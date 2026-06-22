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


class TFF1D(nn.Module):
    """
    1D 版 TFF（来自 包/6_8_TFF.py 的思想，适配时序 B,C,L）
    - depthwise separable Conv1d
    - 融合两支特征（例如季节性与趋势）
    """
    def __init__(self, channels: int):
        super().__init__()
        in_ch = channels
        # 两支拼接后通道数为 2C
        self.catconvA = nn.Sequential(
            nn.Conv1d(2 * in_ch, 2 * in_ch, kernel_size=3, padding=1, groups=2 * in_ch),
            nn.Conv1d(2 * in_ch, in_ch, kernel_size=1, padding=0, groups=1),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=False),
        )
        self.catconvB = nn.Sequential(
            nn.Conv1d(2 * in_ch, 2 * in_ch, kernel_size=3, padding=1, groups=2 * in_ch),
            nn.Conv1d(2 * in_ch, in_ch, kernel_size=1, padding=0, groups=1),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=False),
        )
        self.catconv = nn.Sequential(
            nn.Conv1d(2 * in_ch, 2 * in_ch, kernel_size=3, padding=1, groups=2 * in_ch),
            nn.Conv1d(2 * in_ch, in_ch, kernel_size=1, padding=0, groups=1),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=False),
        )
        self.convA = nn.Conv1d(in_ch, 1, kernel_size=1)
        self.convB = nn.Conv1d(in_ch, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        # xA, xB: (B, C, L)
        x_diff = xA - xB
        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))
        A_weight = self.sigmoid(self.convA(x_diffA))  # (B,1,L)
        B_weight = self.sigmoid(self.convB(x_diffB))  # (B,1,L)
        xA = A_weight * xA
        xB = B_weight * xB
        x = self.catconv(torch.cat([xA, xB], dim=1))
        return x


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
        self.linear_layers = nn.ModuleDict({ str(i): nn.Linear(seq_len, pred_len) for i in range(n_clusters) })

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
    YFT: AdaWaveNet + TFF1D 融合（用 TFF1D 融合季节性预测与趋势预测）
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

        # TFF1D 融合模块（通道数=enc_in）
        self.tff = TFF1D(channels=configs.enc_in)

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len // (2 ** configs.lifting_levels), configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Encoder/Decoder levels
        self.encoder_levels = nn.ModuleList()
        self.linear_levels = nn.ModuleList()
        self.coef_linear_levels = nn.ModuleList()
        self.coef_dec_levels = nn.ModuleList()
        in_planes = configs.enc_in
        input_size = self.seq_len
        expand_ratio = configs.sr_ratio if self.task_name == 'super_resolution' else 1
        for i in range(configs.lifting_levels):
            self.encoder_levels.add_module(f'encoder_level_{i}', AdpWaveletBlock(configs, input_size))
            in_planes *= 1
            input_size = input_size // 2
            self.linear_levels.add_module(f'linear_level_{i}', nn.Sequential(nn.Linear(input_size, input_size * expand_ratio)))
            self.coef_linear_levels.add_module(f'coef_linear_level_{i}', nn.Sequential(nn.Linear(input_size, input_size * expand_ratio)))
            self.coef_dec_levels.add_module(f'coef_dec_level_{i}', nn.Sequential(nn.Linear(input_size, input_size * expand_ratio)))

        self.input_size = input_size

        self.decoder_levels = nn.ModuleList()
        for i in range(configs.lifting_levels - 1, -1, -1):
            self.decoder_levels.add_module(f'decoder_level_{i}', InverseAdpWaveletBlock(configs, input_size=input_size))
            in_planes //= 1
            input_size *= 2

        if self.task_name == 'super_resolution':
            self.lowrank_projection = nn.Linear(configs.d_model, self.pred_len // (2 ** configs.lifting_levels), bias=True)
        else:
            self.lowrank_projection = nn.Linear(configs.d_model, self.seq_len // (2 ** configs.lifting_levels), bias=True)

        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention), configs.d_model, configs.n_heads),
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
        seasonal, trend = self.series_decomp(x_enc.permute(0, 2, 1))  # (B,C,L)
        trend = trend.permute(0, 2, 1)

        # 季节性路径编码/解码
        x_norm, means, stdev = self._norm(seasonal)
        _, _, N = x_norm.shape
        x_dec = self._encode_decode(x_norm)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denorm(dec_out, means, stdev)  # (B, pred, C)

        # 趋势路径（聚类线性）
        trend = self.rev_trend(trend, 'norm')
        trend_out = self.trend_linear(trend.permute(0, 2, 1), self.clusters).permute(0, 2, 1)  # (B, pred, C)
        trend_out = self.rev_trend(trend_out, 'denorm')

        # TFF1D 融合（到 B,C,L）
        s_bcl = dec_out.permute(0, 2, 1)
        t_bcl = trend_out.permute(0, 2, 1)
        fused_bcl = self.tff(s_bcl, t_bcl)
        fused = fused_bcl.permute(0, 2, 1)
        return fused

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        seasonal, trend = self.series_decomp(x_enc.permute(0, 2, 1))
        trend = trend.permute(0, 2, 1)
        x_norm, means, stdev = self._norm(seasonal)
        _, _, N = x_norm.shape
        x_dec = self._encode_decode(x_norm)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denorm(dec_out, means, stdev)
        trend = self.rev_trend(trend, 'norm')
        trend_out = self.trend_linear(trend.permute(0, 2, 1), self.clusters).permute(0, 2, 1)
        trend_out = self.rev_trend(trend_out, 'denorm')
        fused = self.tff(dec_out.permute(0, 2, 1), trend_out.permute(0, 2, 1)).permute(0, 2, 1)
        return fused

    def anomaly_detection(self, x_enc):
        seasonal, trend = self.series_decomp(x_enc.permute(0, 2, 1))
        trend = trend.permute(0, 2, 1)
        x_norm, means, stdev = self._norm(seasonal)
        _, _, N = x_norm.shape
        x_dec = self._encode_decode(x_norm)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denorm(dec_out, means, stdev)
        trend = self.rev_trend(trend, 'norm')
        trend_out = self.trend_linear(trend.permute(0, 2, 1), self.clusters).permute(0, 2, 1)
        trend_out = self.rev_trend(trend_out, 'denorm')
        fused = self.tff(dec_out.permute(0, 2, 1), trend_out.permute(0, 2, 1)).permute(0, 2, 1)
        return fused

    def super_resolution(self, x_enc):
        seasonal, trend = self.series_decomp(x_enc.permute(0, 2, 1))
        trend = trend.permute(0, 2, 1)
        x_norm, means, stdev = self._norm(seasonal)
        _, _, N = x_norm.shape
        x_dec = self._encode_decode(x_norm)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denorm(dec_out, means, stdev)
        trend = self.rev_trend(trend, 'norm')
        trend_out = self.trend_linear(trend.permute(0, 2, 1), self.clusters).permute(0, 2, 1)
        trend_out = self.rev_trend(trend_out, 'denorm')
        fused = self.tff(dec_out.permute(0, 2, 1), trend_out.permute(0, 2, 1)).permute(0, 2, 1)
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
            raise ValueError('classification not implemented for yft')
        if self.task_name == 'super_resolution':
            return self.super_resolution(x_enc)
        return None




