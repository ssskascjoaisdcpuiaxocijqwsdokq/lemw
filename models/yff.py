import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.LiftingScheme import LiftingScheme, InverseLiftingScheme
from layers.Invertible import RevIN


def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result


class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, small_kernel, small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups, bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=small_kernel,
                                          stride=stride, padding=small_kernel // 2, groups=groups, dilation=1, bias=False)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out


class EMA1D(nn.Module):
    """
    1D 版 EMA（改编自 包/2_3_EMAttention.py 的思想）。
    - 分组处理通道后，使用 1x1 与 3x3 分支并进行跨分支的权重交互，生成空间权重以重标定输入。
    输入/输出: (B, C, L)
    """
    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.groups = max(1, min(groups, channels))
        assert channels // self.groups > 0
        c_g = channels // self.groups
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool1d(1)
        self.conv1x1 = nn.Conv1d(c_g, c_g, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv1d(c_g, c_g, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        b, c, l = x.size()
        g = self.groups
        c_g = c // g
        group_x = x.reshape(b * g, c_g, l)
        # 两分支
        x1 = self.conv1x1(group_x)
        x2 = self.conv3x3(group_x)
        # 通道描述符
        x11 = self.softmax(self.agp(x1).reshape(b * g, 1, c_g))  # (b*g,1,c_g)
        x12 = x2.reshape(b * g, c_g, -1)                          # (b*g,c_g,l)
        y1 = torch.matmul(x11, x12)                                # (b*g,1,l)
        x21 = self.softmax(self.agp(x2).reshape(b * g, 1, c_g))    # (b*g,1,c_g)
        x22 = x1.reshape(b * g, c_g, -1)                          # (b*g,c_g,l)
        y2 = torch.matmul(x21, x22)                                # (b*g,1,l)
        weights = (y1 + y2).reshape(b * g, 1, l).sigmoid()         # (b*g,1,l)
        out = (group_x * weights).reshape(b, c, l)
        return out


class MultiScaleDilatedConv1D(nn.Module):
    """
    多尺度膨胀卷积 1D：并联不同 dilation 的 Conv1d 后聚合，适合趋势/长依赖增强。
    输入/输出: (B, C, L)
    """
    def __init__(self, channels: int, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=3, padding=d, dilation=d, groups=channels)
            for d in dilations
        ])
        self.aggr = nn.Conv1d(len(dilations) * channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(x) for b in self.branches]
        x_cat = torch.cat(outs, dim=1)
        return self.aggr(x_cat)


class CoordAtt1D(nn.Module):
    """
    1D 版 Coordinate Attention（改编自 包/1_5_CoordAttention.py 思想）。
    同时编码通道与位置信息，生成逐位置的通道权重。
    输入/输出: (B, C, L)
    """
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool_l = nn.AdaptiveAvgPool1d(1)
        self.conv_reduce = nn.Conv1d(channels, hidden, kernel_size=1)
        self.act = nn.ReLU(inplace=False)
        self.conv_expand = nn.Conv1d(hidden, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        l_enc = self.pool_l(x)                 # (B,C,1)
        fused = self.conv_reduce(l_enc)        # (B,H,1)
        fused = self.act(fused)
        attn = self.conv_expand(fused)         # (B,C,1)
        attn = self.sigmoid(attn)              # (B,C,1)
        return x * attn


class ReluLinearAttention1D(nn.Module):
    """
    1D ReLU 线性注意力（灵感自 包/4_2_Relu_Linear_Attention.py）。
    在 (B,L,C) 计算 A = softmax(relu(Q) @ relu(K)^T / L)，Y = A @ V (V=K)。
    输入 (B,C,L)，输出 (B,C,L)。
    """
    def __init__(self, channels: int):
        super().__init__()
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        # x_bcl: (B, C, L)
        x = x_bcl.permute(0, 2, 1)  # (B, L, C)
        B, L, C = x.shape
        Q = F.relu(self.to_q(x))
        K = F.relu(self.to_k(x))
        attn = torch.matmul(Q, K.transpose(1, 2)) / max(L, 1)
        attn = F.softmax(attn, dim=-1)
        V = K
        Y = torch.matmul(attn, V)
        Y = self.proj(Y)
        return Y.permute(0, 2, 1)


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2))
        end = x[:, :, -1:].repeat(1, 1, math.floor((self.kernel_size - 1) // 2))
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x


class moving_avg_imputation(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg_imputation, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        num_channels = x.shape[1]
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2))
        end = x[:, :, -1:].repeat(1, 1, math.floor((self.kernel_size - 1) // 2))
        x_padded = torch.cat([front, x, end], dim=-1)
        non_zero_mask = x_padded != 0
        weight = torch.ones((1, num_channels, self.kernel_size), device=x_padded.device)
        window_sum = F.conv1d(x_padded, weight=weight, stride=self.stride)
        window_count = F.conv1d(non_zero_mask.float(), weight=weight, stride=self.stride)
        window_count = torch.clamp(window_count, min=1)
        moving_avg = window_sum / window_count
        return moving_avg


class series_decomp(nn.Module):
    def __init__(self, kernel_size=24, stride=1, imputation=False):
        super(series_decomp, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.moving_avg = moving_avg(kernel_size, stride=stride) if not imputation else moving_avg_imputation(self.kernel_size, self.stride)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class AdpWaveletBlock(nn.Module):
    def __init__(self, configs, input_size):
        super(AdpWaveletBlock, self).__init__()
        self.regu_details = configs.regu_details
        self.regu_approx = configs.regu_approx
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()
        self.wavelet = LiftingScheme(configs.enc_in, k_size=configs.lifting_kernel_size, input_size=input_size)
        self.norm_x = normalization(configs.enc_in)
        self.norm_d = normalization(configs.enc_in)

    def forward(self, x):
        (c, d) = self.wavelet(x)
        x = c
        r = None
        if (self.regu_approx + self.regu_details) != 0.0:
            if self.regu_details:
                rd = self.regu_details * d.abs().mean()
            if self.regu_approx:
                rc = self.regu_approx * torch.dist(c.mean(), x.mean(), p=2)
            if self.regu_approx == 0.0:
                r = rd
            elif self.regu_details == 0.0:
                r = rc
            else:
                r = rd + rc
        x = self.norm_x(x)
        d = self.norm_d(d)
        return x, r, d


class InverseAdpWaveletBlock(nn.Module):
    def __init__(self, configs, input_size):
        super(InverseAdpWaveletBlock, self).__init__()
        self.inverse_wavelet = InverseLiftingScheme(configs.enc_in, input_size=input_size, kernel_size=configs.lifting_kernel_size)

    def forward(self, c, d):
        reconstructed = self.inverse_wavelet(c, d)
        return reconstructed


class ClusteredLinear(nn.Module):
    def __init__(self, n_clusters, enc_in, seq_len, pred_len):
        super().__init__()
        self.n_clusters = n_clusters
        self.enc_in = enc_in
        self.linear_layers = nn.ModuleDict({str(i): nn.Linear(seq_len, pred_len) for i in range(n_clusters)})

    def forward(self, x, clusters):
        output = []
        assert self.enc_in == len(clusters)
        for channel in range(self.enc_in):
            cluster_id = str(clusters[channel].item())
            channel_data = x[:, channel, :].unsqueeze(1)
            transformed_channel = self.linear_layers[cluster_id](channel_data)
            output.append(transformed_channel)
        output = torch.concat(output, dim=1)
        return output


class Model(nn.Module):
    """
    YFF: AdaWaveNet + ModernTCNBlock1D + LSKModule1D
    - ModernTCN 大核卷积用于趋势分量的时间建模
    - LSK 多尺度选择性核用于季节性分量的特征增强
    - 直接相加融合季节性和趋势预测
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'super_resolution':
            self.seq_len = self.seq_len // configs.sr_ratio
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.kmeans = KMeans(n_clusters=configs.n_clusters)
        self.series_decomp = series_decomp(imputation=self.task_name == 'imputation')
        self.rev_seasonal = RevIN(configs.enc_in)
        self.rev_trend = RevIN(configs.enc_in)
        self.trend_linear = ClusteredLinear(configs.n_clusters, configs.enc_in, self.seq_len, configs.pred_len)

        # 趋势：ReLU 线性注意力
        self.trend_rla = ReluLinearAttention1D(channels=configs.enc_in)

        # 季节性：CoordAtt1D 注意力增强
        self.seasonal_coord = CoordAtt1D(channels=configs.enc_in, reduction=getattr(configs, 'coord_reduction', 32))

        # 计算可用的小波层数，避免序列过短导致错误
        self.actual_levels = self._compute_actual_levels(self.seq_len, getattr(configs, 'lifting_levels', 3), getattr(configs, 'lifting_kernel_size', 7))

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len // (2 ** self.actual_levels),
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # Encoder/Decoder levels
        self.encoder_levels = nn.ModuleList()
        self.linear_levels = nn.ModuleList()
        self.coef_linear_levels = nn.ModuleList()
        self.coef_dec_levels = nn.ModuleList()
        in_planes = configs.enc_in
        input_size = self.seq_len
        expand_ratio = configs.sr_ratio if self.task_name == 'super_resolution' else 1
        
        for i in range(self.actual_levels):
            self.encoder_levels.add_module(f'encoder_level_{i}', AdpWaveletBlock(configs, input_size))
            in_planes *= 1
            input_size = input_size // 2
            self.linear_levels.add_module(f'linear_level_{i}', nn.Sequential(nn.Linear(input_size, input_size * expand_ratio)))
            self.coef_linear_levels.add_module(f'coef_linear_level_{i}', nn.Sequential(nn.Linear(input_size, input_size * expand_ratio)))
            self.coef_dec_levels.add_module(f'coef_dec_level_{i}', nn.Sequential(nn.Linear(input_size, input_size * expand_ratio)))

        self.input_size = input_size

        self.decoder_levels = nn.ModuleList()
        for i in range(self.actual_levels - 1, -1, -1):
            self.decoder_levels.add_module(f'decoder_level_{i}', InverseAdpWaveletBlock(configs, input_size=input_size))
            in_planes //= 1
            input_size *= 2

        if self.task_name == 'super_resolution':
            self.lowrank_projection = nn.Linear(configs.d_model, self.pred_len // (2 ** self.actual_levels), bias=True)
        else:
            self.lowrank_projection = nn.Linear(configs.d_model, self.seq_len // (2 ** self.actual_levels), bias=True)

        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                    configs.d_model,
                    configs.n_heads
                ),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for _ in range(configs.e_layers)
        ], norm_layer=torch.nn.LayerNorm(configs.d_model))

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(self.seq_len, configs.pred_len, bias=True)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(self.seq_len, self.seq_len, bias=True)
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(self.seq_len, self.seq_len, bias=True)
        elif self.task_name == 'super_resolution':
            self.projection = nn.Linear(configs.pred_len, configs.pred_len, bias=True)

        self.register_buffer('clusters', None)

        # 轻量正则
        self.seasonal_dropout = nn.Dropout(p=getattr(configs, 'dropout', 0.1))
        self.trend_dropout = nn.Dropout(p=getattr(configs, 'dropout', 0.1))

        # 权重初始化
        self._init_weights()

    def _compute_actual_levels(self, seq_len: int, desired_levels: int, ksize: int) -> int:
        levels = 0
        current = seq_len
        # 保证每一层输入长度足够，并且至少能再对半一次
        for _ in range(desired_levels):
            if current < max(ksize, 4):
                break
            levels += 1
            current = current // 2
        return max(levels, 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def _norm(self, x_bcl: torch.Tensor):
        x_blc = x_bcl.permute(0, 2, 1)
        means = x_blc.mean(1, keepdim=True).detach()
        x_blc = x_blc - means
        stdev = torch.sqrt(torch.var(x_blc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_blc = x_blc / stdev
        x_bcl_norm = x_blc.permute(0, 2, 1)
        return x_bcl_norm, means, stdev

    def _denorm(self, dec_out, means, stdev):
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def _encode_decode(self, x_enc):
        # x_enc: (B, C, L)
        encoded_coefficients, x_embedding_levels, coef_embedding_levels = [], [], []
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
        x_enc = x_enc.permute(0, 2, 1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        for dec, x_emb_level, coef_emb_level, c_linear in zip(self.decoder_levels, x_embedding_levels[::-1], coef_embedding_levels[::-1], self.coef_dec_levels[::-1]):
            details = encoded_coefficients.pop()
            details = coef_emb_level + c_linear(details)
            x_dec = x_dec + x_emb_level
            x_dec = dec(x_dec, details)
        return x_dec

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, clusters):
        # 分解
        seasonal, trend = self.series_decomp(x_enc.permute(0, 2, 1))  # (B, C, L)
        trend = trend.permute(0, 2, 1)  # (B, L, C)

        # 季节性路径：LSK增强 + 小波编码/解码
        seasonal_enhanced = self.seasonal_coord(seasonal)  # (B, C, L)
        seasonal_enhanced = self.seasonal_dropout(seasonal_enhanced)
        x_norm, means, stdev = self._norm(seasonal_enhanced)
        _, _, N = x_norm.shape
        x_dec = self._encode_decode(x_norm)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        seasonal_pred = self._denorm(dec_out, means, stdev)  # (B, pred_len, C)

        # 趋势路径：ModernTCN + 聚类线性
        trend = self.rev_trend(trend, 'norm')
        trend_bcl = trend.permute(0, 2, 1)  # (B, C, L)
        trend_enhanced = self.trend_rla(trend_bcl)  # (B, C, L)
        trend_enhanced = self.trend_dropout(trend_enhanced)
        trend_out = self.trend_linear(trend_enhanced, self.clusters).permute(0, 2, 1)  # (B, pred_len, C)
        trend_pred = self.rev_trend(trend_out, 'denorm')

        # 直接相加融合
        fused = seasonal_pred + trend_pred
        return fused

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        seasonal, trend = self.series_decomp(x_enc.permute(0, 2, 1))
        trend = trend.permute(0, 2, 1)
        seasonal_enhanced = self.seasonal_coord(seasonal)
        x_norm, means, stdev = self._norm(seasonal_enhanced)
        _, _, N = x_norm.shape
        x_dec = self._encode_decode(x_norm)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        seasonal_pred = self._denorm(dec_out, means, stdev)
        trend = self.rev_trend(trend, 'norm')
        trend_enhanced = self.trend_rla(trend.permute(0, 2, 1))
        trend_out = self.trend_linear(trend_enhanced, self.clusters).permute(0, 2, 1)
        trend_pred = self.rev_trend(trend_out, 'denorm')
        fused = seasonal_pred + trend_pred
        return fused

    def anomaly_detection(self, x_enc):
        seasonal, trend = self.series_decomp(x_enc.permute(0, 2, 1))
        trend = trend.permute(0, 2, 1)
        seasonal_enhanced = self.seasonal_coord(seasonal)
        x_norm, means, stdev = self._norm(seasonal_enhanced)
        _, _, N = x_norm.shape
        x_dec = self._encode_decode(x_norm)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        seasonal_pred = self._denorm(dec_out, means, stdev)
        trend = self.rev_trend(trend, 'norm')
        trend_enhanced = self.trend_rla(trend.permute(0, 2, 1))
        trend_out = self.trend_linear(trend_enhanced, self.clusters).permute(0, 2, 1)
        trend_pred = self.rev_trend(trend_out, 'denorm')
        fused = seasonal_pred + trend_pred
        return fused

    def super_resolution(self, x_enc):
        seasonal, trend = self.series_decomp(x_enc.permute(0, 2, 1))
        trend = trend.permute(0, 2, 1)
        seasonal_enhanced = self.seasonal_eaa(seasonal)
        x_norm, means, stdev = self._norm(seasonal_enhanced)
        _, _, N = x_norm.shape
        x_dec = self._encode_decode(x_norm)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        seasonal_pred = self._denorm(dec_out, means, stdev)
        trend = self.rev_trend(trend, 'norm')
        trend_enhanced = self.trend_eca(trend.permute(0, 2, 1))
        trend_out = self.trend_linear(trend_enhanced, self.clusters).permute(0, 2, 1)
        trend_pred = self.rev_trend(trend_out, 'denorm')
        fused = seasonal_pred + trend_pred
        return fused

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, C = x_enc.shape
        x_cluster = x_enc.permute(2, 0, 1).view(C, B * L)
        if self.clusters is None:
            clusters = self.kmeans.fit_predict(x_cluster)
            self.clusters = clusters
        else:
            clusters = self.clusters
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, clusters)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            raise ValueError('classification not implemented for yff')
        if self.task_name == 'super_resolution':
            return self.super_resolution(x_enc)
        return None
