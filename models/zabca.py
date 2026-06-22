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
# import pywt  # 移除pywt依赖，使用PyTorch内置方法

def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)

class WTFD1D(nn.Module):
    """
    1D小波变换高低频特征分解模块 - 适配时序数据
    基于WTFD模块，针对1D时序数据进行优化
    """
    def __init__(self, in_ch, out_ch, wavelet='haar'):
        super(WTFD1D, self).__init__()
        self.wavelet = wavelet
        
        # 高频特征处理 - 3个高频分量的融合
        self.high_freq_conv = nn.Sequential(
            nn.Conv1d(in_ch * 3, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
        )
        
        # 低频特征输出
        self.low_freq_out = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        
        # 高频特征输出
        self.high_freq_out = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        
    def dwt_1d(self, x):
        """
        简化的1D小波变换 - 使用PyTorch内置方法
        """
        # x: (B, C, L)
        B, C, L = x.shape
        
        # 简化的小波变换：使用平均池化和差分近似
        # 低频分量：平均池化
        low_freq = F.avg_pool1d(x, kernel_size=2, stride=2)
        
        # 高频分量：差分操作
        # 先进行填充以保持尺寸
        x_padded = F.pad(x, (1, 0), mode='reflect')
        high_freq = x_padded[:, :, 1:] - x_padded[:, :, :-1]  # 差分
        high_freq = F.avg_pool1d(high_freq, kernel_size=2, stride=2)  # 下采样
        
        return low_freq, high_freq
    
    def forward(self, x):
        # x: (B, C, L)
        low_freq, high_freq = self.dwt_1d(x)
        
        # 创建多个高频分量（模拟2D的HL, LH, HH）
        # 通过不同的卷积核模拟不同方向的高频信息
        high_1 = high_freq  # 原始高频
        high_2 = F.avg_pool1d(high_freq, kernel_size=2, stride=1, padding=1)[:, :, :high_freq.size(-1)]  # 平滑高频
        high_3 = F.max_pool1d(high_freq, kernel_size=2, stride=1, padding=1)[:, :, :high_freq.size(-1)]  # 尖锐高频
        
        # 融合三个高频分量
        high_combined = torch.cat([high_1, high_2, high_3], dim=1)  # (B, 3*C, L//2)
        high_processed = self.high_freq_conv(high_combined)  # (B, C, L//2)
        
        # 输出处理
        low_out = self.low_freq_out(low_freq)   # (B, out_ch, L//2)
        high_out = self.high_freq_out(high_processed)  # (B, out_ch, L//2)
        
        return low_out, high_out

class MSDI1D(nn.Module):
    """
    1D多尺度特征融合模块 - 适配时序数据
    基于MSDI模块，针对1D时序数据进行优化
    """
    def __init__(self, channels):
        super(MSDI1D, self).__init__()
        
        # 为每个输入通道创建1D卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(c, channels[0], kernel_size=3, stride=1, padding=1) 
            for c in channels
        ])
        
    def forward(self, x_list):
        # x_list: 包含不同尺度特征的列表
        if not x_list:
            return None
            
        # 以第一个特征为目标尺寸
        target_size = x_list[0].shape[-1]
        ans = torch.ones_like(x_list[0])
        
        for i, x in enumerate(x_list):
            # 尺寸对齐
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool1d(x, target_size)
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=target_size, mode='linear', align_corners=True)
            
            # Hadamard积融合
            ans = ans * self.convs[i](x)
            
        return ans

class WaveletMultiScaleEncoder(nn.Module):
    """
    小波多尺度编码器 - 融合WTFD和MSDI的优势
    """
    def __init__(self, configs):
        super(WaveletMultiScaleEncoder, self).__init__()
        
        # 多个WTFD层用于不同尺度的小波分解
        self.wtfd_layers = nn.ModuleList([
            WTFD1D(configs.d_model, configs.d_model) for _ in range(3)
        ])
        
        # MSDI融合层
        self.msdi_low = MSDI1D([configs.d_model] * 3)
        self.msdi_high = MSDI1D([configs.d_model] * 3)
        
        # 特征增强
        self.feature_enhance = nn.Sequential(
            nn.Conv1d(configs.d_model * 2, configs.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(configs.d_model),
            nn.GELU(),
            nn.Conv1d(configs.d_model, configs.d_model, kernel_size=1),
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            configs.d_model, 
            num_heads=configs.n_heads, 
            batch_first=True, 
            dropout=0.1
        )
        
        # 标准Transformer编码器
        self.transformer_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), 
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # 自适应融合权重
        self.fusion_weight = nn.Parameter(torch.tensor([0.4, 0.6]))  # [wavelet, transformer]
        self.final_norm = nn.LayerNorm(configs.d_model)
        
    def forward(self, x, attn_mask=None):
        # x: (B, L, C)
        B, L, C = x.shape
        x_conv = x.permute(0, 2, 1)  # (B, C, L) for conv operations
        
        # 多尺度小波分解
        low_features = []
        high_features = []
        
        current_x = x_conv
        for wtfd in self.wtfd_layers:
            low, high = wtfd(current_x)
            low_features.append(low)
            high_features.append(high)
            # 下采样用于下一层
            current_x = F.avg_pool1d(current_x, kernel_size=2, stride=2)
            if current_x.size(-1) < 4:  # 防止过度下采样
                break
        
        # 多尺度特征融合
        if len(low_features) > 1:
            fused_low = self.msdi_low(low_features)
            fused_high = self.msdi_high(high_features)
        else:
            fused_low = low_features[0] if low_features else x_conv
            fused_high = high_features[0] if high_features else x_conv
        
        # 特征组合和增强
        if fused_low.size(-1) != fused_high.size(-1):
            target_size = min(fused_low.size(-1), fused_high.size(-1))
            fused_low = F.adaptive_avg_pool1d(fused_low, target_size)
            fused_high = F.adaptive_avg_pool1d(fused_high, target_size)
        
        combined = torch.cat([fused_low, fused_high], dim=1)  # (B, 2*C, L')
        enhanced = self.feature_enhance(combined)  # (B, C, L')
        
        # 恢复到原始长度
        if enhanced.size(-1) != L:
            enhanced = F.interpolate(enhanced, size=L, mode='linear', align_corners=True)
        
        enhanced = enhanced.permute(0, 2, 1)  # (B, L, C)
        
        # 注意力增强
        attn_out, _ = self.attention(enhanced, enhanced, enhanced)
        wavelet_out = enhanced + attn_out
        
        # 标准Transformer处理
        transformer_out, attns = self.transformer_encoder(x, attn_mask=attn_mask)
        
        # 自适应融合
        weights = F.softmax(self.fusion_weight, dim=0)
        output = weights[0] * wavelet_out + weights[1] * transformer_out
        
        # 最终归一化
        output = self.final_norm(output)
        
        return output, attns

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # x - B, C, L
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2))
        end = x[:, :, -1:].repeat(1, 1, math.floor((self.kernel_size - 1) // 2))
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size=24, stride=1):
        super(series_decomp, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.moving_avg = moving_avg(kernel_size, stride=stride)

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
        
        # Define a linear layer for each cluster using ModuleDict
        self.linear_layers = nn.ModuleDict({
            str(cluster_id): nn.Linear(seq_len, pred_len) for cluster_id in range(n_clusters)
        })
        
    def forward(self, x, clusters):
        output = []
        assert self.enc_in == len(clusters)
        for channel in range(self.enc_in):
            cluster_id = str(clusters[channel].item())
            channel_data = x[:, channel, :].unsqueeze(1)  # Reshape to keep the channel dimension
            transformed_channel = self.linear_layers[cluster_id](channel_data)
            output.append(transformed_channel)
        output = torch.concat(output, dim=1)
        
        return output

class Model(nn.Module):
    """
    ZABCA: AdaWaveNet + WTFD1D + MSDI1D
    融合小波高低频分解和多尺度特征融合的时序预测模型
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
        
         # Construct the levels recursively (encoder)
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
                AdpWaveletBlock(configs, input_size)
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
        
        # Construct the levels recursively (decoder)
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

        # 小波多尺度编码器 (融合WTFD和MSDI)
        self.encoder = WaveletMultiScaleEncoder(configs)
        
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
        x_enc /= stdev
        _, _, N = x_enc.shape

        moving_mean = self.rev_trend(moving_mean, 'norm')

        x_enc = x_enc.permute(0,2,1)
        encoded_coefficients = []
        x_embedding_levels = []
        coef_embedding_levels = []
        # Encoding
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
        
        # Embedding
        x_enc = x_enc.permute(0,2,1)
        enc_out = self.enc_embedding(x_enc, None)
        
        # 小波多尺度编码器处理
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
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        
        return None