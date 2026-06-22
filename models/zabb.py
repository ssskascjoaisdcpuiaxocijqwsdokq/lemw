import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from layers.LiftingScheme import LiftingScheme, InverseLiftingScheme
from layers.Invertible import RevIN

def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)

# ==================== PKIM和CAA相关模块 ====================
def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

def autopad(kernel_size: int, padding: int = None, dilation: int = 1):
    assert kernel_size % 2 == 1, 'if use autopad, kernel size must be odd'
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1
    if padding is None:
        padding = kernel_size // 2
    return padding

def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value

class GSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool1d(1)  # 适配1D

    def forward(self, x):
        return x * torch.sigmoid(self.adpool(x))

# 适配1D的CAA模块
class CAA1D(nn.Module):
    """Context Anchor Attention for 1D sequences"""
    def __init__(self, channels: int, kernel_size: int = 11):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(7, 1, 3)
        self.conv1 = nn.Sequential(
            nn.Conv1d(channels, channels, 1, 1, 0),
            nn.BatchNorm1d(channels),
            nn.SiLU()
        )
        self.h_conv = nn.Conv1d(channels, channels, kernel_size, 1, kernel_size // 2, groups=channels)
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels, channels, 1, 1, 0),
            nn.BatchNorm1d(channels),
            nn.SiLU()
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, L]
        attn_factor = self.act(self.conv2(self.h_conv(self.conv1(self.avg_pool(x)))))
        return attn_factor

# 适配1D的PKIM模块
class PKIModule1D(nn.Module):
    """1D版本的Poly Kernel Inception Module"""
    def __init__(self, in_channels: int, out_channels: int = None, 
                 kernel_sizes: tuple = (3, 5, 7, 9, 11), expansion: float = 1.0,
                 with_caa: bool = True, caa_kernel_size: int = 11):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.pre_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1, 1, 0),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU()
        )

        # 多尺度深度卷积
        self.dw_convs = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, k, 1, autopad(k), groups=hidden_channels)
            for k in kernel_sizes
        ])
        
        self.pw_conv = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1, 1, 0),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU()
        )

        if with_caa:
            self.caa_factor = CAA1D(hidden_channels, caa_kernel_size)
        else:
            self.caa_factor = None

        self.post_conv = nn.Sequential(
            nn.Conv1d(hidden_channels, out_channels, 1, 1, 0),
            nn.BatchNorm1d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        # x: [B, C, L]
        x = self.pre_conv(x)
        y = x.clone()
        
        # 多尺度特征融合
        multi_scale_features = [conv(x) for conv in self.dw_convs]
        x = sum(multi_scale_features)
        x = self.pw_conv(x)
        
        if self.caa_factor is not None:
            y = self.caa_factor(y)
            x = x * y
        
        x = x + y  # 残差连接
        x = self.post_conv(x)
        return x

# ==================== HWD小波下采样适配1D ====================
class HWD1D(nn.Module):
    """1D版本的Haar Wavelet Downsampling"""
    def __init__(self, in_ch, out_ch):
        super(HWD1D, self).__init__()
        # 使用1D卷积模拟Haar小波变换
        self.low_pass = nn.Conv1d(in_ch, in_ch, kernel_size=2, stride=2, groups=in_ch, bias=False)
        self.high_pass = nn.Conv1d(in_ch, in_ch, kernel_size=2, stride=2, groups=in_ch, bias=False)
        
        # 初始化Haar小波滤波器
        with torch.no_grad():
            # 低通滤波器 [1, 1] / sqrt(2)
            self.low_pass.weight.fill_(1.0 / math.sqrt(2))
            # 高通滤波器 [1, -1] / sqrt(2)
            self.high_pass.weight[:, :, 0] = 1.0 / math.sqrt(2)
            self.high_pass.weight[:, :, 1] = -1.0 / math.sqrt(2)
        
        self.conv_bn_relu = nn.Sequential(
            nn.Conv1d(in_ch * 2, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # x: [B, C, L]
        low = self.low_pass(x)
        high = self.high_pass(x)
        x = torch.cat([low, high], dim=1)
        x = self.conv_bn_relu(x)
        return x

# ==================== AdaWaveNet核心模块 ====================
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2))
        end = x[:, :, -1:].repeat(1, 1, math.floor((self.kernel_size - 1) // 2))
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size=24, stride=1):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=stride)

    def forward(self, x):
        moving_mean = self.moving_avg(x) 
        res = x - moving_mean
        return res, moving_mean

class EnhancedAdpWaveletBlock(nn.Module):
    """增强的自适应小波块，融合PKIM和HWD"""
    def __init__(self, configs, input_size):
        super(EnhancedAdpWaveletBlock, self).__init__()
        self.regu_details = configs.regu_details
        self.regu_approx = configs.regu_approx
        
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        # 原始小波变换
        self.wavelet = LiftingScheme(configs.enc_in, k_size=configs.lifting_kernel_size, input_size=input_size)
        
        # PKIM多尺度特征提取
        self.pkim = PKIModule1D(configs.enc_in, configs.enc_in, with_caa=True)
        
        # HWD小波下采样（用于特征增强）
        self.hwd_enhance = HWD1D(configs.enc_in, configs.enc_in)
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(configs.enc_in * 2, configs.enc_in, 1),
            nn.BatchNorm1d(configs.enc_in),
            nn.SiLU()
        )
        
        self.norm_x = normalization(configs.enc_in)
        self.norm_d = normalization(configs.enc_in)

    def forward(self, x):
        # x: [B, C, L]
        # 原始小波分解
        (c, d) = self.wavelet(x)
        
        # PKIM多尺度特征提取
        pkim_features = self.pkim(x)
        
        # HWD增强（如果序列长度足够）
        if x.size(-1) >= 2:
            hwd_features = self.hwd_enhance(x)
            # 上采样回原始长度
            if hwd_features.size(-1) != x.size(-1):
                hwd_features = F.interpolate(hwd_features, size=x.size(-1), mode='linear', align_corners=False)
        else:
            hwd_features = x
        
        # 确保PKIM特征与原始输入维度匹配
        if pkim_features.size(-1) != x.size(-1):
            pkim_features = F.interpolate(pkim_features, size=x.size(-1), mode='linear', align_corners=False)
        
        # 特征融合
        enhanced_features = torch.cat([pkim_features, hwd_features], dim=1)
        enhanced_features = self.feature_fusion(enhanced_features)
        
        # 将增强特征与小波系数结合
        # 确保维度匹配
        if enhanced_features.size(-1) != c.size(-1):
            enhanced_features = F.interpolate(enhanced_features, size=c.size(-1), mode='linear', align_corners=False)
        x = c + 0.1 * enhanced_features  # 残差连接
        
        r = None
        if(self.regu_approx + self.regu_details != 0.0):
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
        
        self.linear_layers = nn.ModuleDict({
            str(cluster_id): nn.Linear(seq_len, pred_len) for cluster_id in range(n_clusters)
        })
        
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
    ZABB: 融合PKIM多尺度特征提取、CAA注意力、HWD小波下采样与AdaWaveNet的增强模型
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
        self.series_decomp = series_decomp()
        self.rev_seasonal = RevIN(configs.enc_in)
        self.rev_trend = RevIN(configs.enc_in)
        self.trend_linear = ClusteredLinear(configs.n_clusters, configs.enc_in, self.seq_len, configs.pred_len)
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len // (2 ** configs.lifting_levels), configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # 构建增强的编码器层
        self.encoder_levels = nn.ModuleList()
        self.linear_levels = nn.ModuleList()
        self.coef_linear_levels = nn.ModuleList()
        self.coef_dec_levels = nn.ModuleList()
        in_planes = configs.enc_in
        input_size = self.seq_len
        
        if self.task_name == "super_resolution":
            expand_ratio = configs.sr_ratio
        else:
            expand_ratio = 1
        
        for i in range(configs.lifting_levels):
            self.encoder_levels.add_module(
                'encoder_level_'+str(i),
                EnhancedAdpWaveletBlock(configs, input_size)  # 使用增强版本
            )
            in_planes *= 1
            input_size = input_size // 2 
            self.linear_levels.add_module(
                'linear_level_'+str(i),
                nn.Sequential(
                    nn.Linear(input_size, input_size * expand_ratio),
                )
            )
            self.coef_linear_levels.add_module(
                'linear_level_'+str(i),
                nn.Sequential(
                    nn.Linear(input_size, input_size * expand_ratio),
                )
            )
            self.coef_dec_levels.add_module(
                'linear_level_'+str(i),
                nn.Sequential(
                    nn.Linear(input_size, input_size * expand_ratio),
                )
            )

        self.input_size = input_size
        
        # 构建解码器层
        self.decoder_levels = nn.ModuleList()
        for i in range(configs.lifting_levels-1, -1, -1):
            self.decoder_levels.add_module(
                'decoder_level_'+str(i),
                InverseAdpWaveletBlock(configs, input_size=input_size)
            )
            in_planes //= 1
            input_size *= 2
        
        if self.task_name == "super_resolution":
            self.lowrank_projection = nn.Linear(configs.d_model, self.pred_len // (2 ** configs.lifting_levels), bias=True)
        else:
            self.lowrank_projection = nn.Linear(configs.d_model, self.seq_len // (2 ** configs.lifting_levels), bias=True)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(self.seq_len, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(self.seq_len, self.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(self.seq_len, self.seq_len, bias=True)
        if self.task_name == 'super_resolution':
            self.projection = nn.Linear(configs.pred_len, configs.pred_len, bias=True)

        self.register_buffer('clusters', None)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, clusters):
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        
        # Normalization from Non-stationary Transformer
        x_enc = x.permute(0,2,1)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        _, _, N = x_enc.shape

        moving_mean = self.rev_trend(moving_mean, 'norm')
        x_enc = x_enc.permute(0,2,1)
        encoded_coefficients = []
        x_embedding_levels = []
        coef_embedding_levels = []
        
        # Encoding with enhanced wavelet blocks
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
            
        # Embedding
        x_enc = x_enc.permute(0,2,1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        
        # Decoding
        for dec, x_emb_level, coef_emb_level, c_linear in zip(self.decoder_levels, x_embedding_levels[::-1], coef_embedding_levels[::-1], self.coef_dec_levels[::-1]):
            details = encoded_coefficients.pop()
            details = coef_emb_level + c_linear(details)
            x_dec = x_dec + x_emb_level
            x_dec = dec(x_dec, details)
            
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        moving_mean_out = self.trend_linear(moving_mean.permute(0,2,1), self.clusters).permute(0,2,1)
        moving_mean_out = self.rev_trend(moving_mean_out, 'denorm')
        
        dec_out = dec_out + moving_mean_out
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        
        x_enc = x.permute(0,2,1)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        _, _, N = x_enc.shape

        moving_mean = self.rev_trend(moving_mean, 'norm')
        x_enc = x_enc.permute(0,2,1)
        encoded_coefficients = []
        x_embedding_levels = []
        coef_embedding_levels = []
        
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
            
        x_enc = x_enc.permute(0,2,1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        
        for dec, x_emb_level, coef_emb_level, c_linear in zip(self.decoder_levels, x_embedding_levels[::-1], coef_embedding_levels[::-1], self.coef_dec_levels[::-1]):
            details = encoded_coefficients.pop()
            details = coef_emb_level + c_linear(details)
            x_dec = x_dec + x_emb_level
            x_dec = dec(x_dec, details)
            
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        moving_mean_out = self.trend_linear(moving_mean.permute(0,2,1), self.clusters).permute(0,2,1)
        moving_mean_out = self.rev_trend(moving_mean_out, 'denorm')
        
        dec_out = dec_out + moving_mean_out
        return dec_out

    def anomaly_detection(self, x_enc):
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        
        x_enc = x.permute(0,2,1)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        _, _, N = x_enc.shape

        moving_mean = self.rev_trend(moving_mean, 'norm')
        x_enc = x_enc.permute(0,2,1)
        encoded_coefficients = []
        x_embedding_levels = []
        coef_embedding_levels = []
        
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
            
        x_enc = x_enc.permute(0,2,1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        
        for dec, x_emb_level, coef_emb_level, c_linear in zip(self.decoder_levels, x_embedding_levels[::-1], coef_embedding_levels[::-1], self.coef_dec_levels[::-1]):
            details = encoded_coefficients.pop()
            details = coef_emb_level + c_linear(details)
            x_dec = x_dec + x_emb_level
            x_dec = dec(x_dec, details)
            
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        moving_mean_out = self.trend_linear(moving_mean.permute(0,2,1), self.clusters).permute(0,2,1)
        moving_mean_out = self.rev_trend(moving_mean_out, 'denorm')
        
        dec_out = dec_out + moving_mean_out
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, C = x_enc.shape
        x_cluster = x_enc.permute(2,0,1).view(C, B * L)
        if self.clusters is None:
            clusters = self.kmeans.fit_predict(x_cluster)
            self.clusters = clusters
        else:
            clusters = self.clusters
            
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, clusters)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        return None
