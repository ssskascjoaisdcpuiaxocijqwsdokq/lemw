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

class SpikeConv1D(nn.Module):
    """
    1D Spike Convolution adapted for time series
    """
    def __init__(self, conv, step=2):
        super(SpikeConv1D, self).__init__()
        self.conv = conv
        self.step = step

    def forward(self, x):
        # x: (B, C, L) or (step, B, C, L)
        if len(x.shape) == 3:
            x = x.unsqueeze(0).repeat(self.step, 1, 1, 1)  # (step, B, C, L)
        
        outputs = []
        for t in range(self.step):
            out = self.conv(x[t])
            outputs.append(out)
        
        return torch.stack(outputs, dim=0)  # (step, B, C_out, L_out)

class SpikePool1D(nn.Module):
    """
    1D Spike Pooling adapted for time series
    """
    def __init__(self, pool, step=2):
        super(SpikePool1D, self).__init__()
        self.pool = pool
        self.step = step

    def forward(self, x):
        # x: (B, C, L) or (step, B, C, L)
        if len(x.shape) == 3:
            x = x.unsqueeze(0).repeat(self.step, 1, 1, 1)  # (step, B, C, L)
        
        outputs = []
        for t in range(self.step):
            out = self.pool(x[t])
            outputs.append(out)
        
        return torch.stack(outputs, dim=0)  # (step, B, C_out, L_out)

class myBatchNorm1d(nn.Module):
    """
    Temporal BatchNorm for spike sequences
    """
    def __init__(self, bn, step=2):
        super(myBatchNorm1d, self).__init__()
        self.bn = bn
        self.step = step

    def forward(self, x):
        # x: (step, B, C, L)
        outputs = []
        for t in range(self.step):
            out = self.bn(x[t])
            outputs.append(out)
        
        return torch.stack(outputs, dim=0)

class DctSpatialLIF1D(nn.Module):
    """
    DCT Spatial LIF adapted for 1D time series from 1.py
    """
    def __init__(self, step=2, channel=64, length=96, freq_num=32, reduction=16):
        super(DctSpatialLIF1D, self).__init__()
        self.step = step
        self.channel = channel
        self.length = length
        self.freq_num = min(freq_num, length // 2)
        
        # DCT频域处理 - 基于1.py的设计
        self.dct_conv = nn.Conv1d(channel, channel // reduction, 1, bias=False)
        self.freq_conv = nn.Conv1d(channel // reduction, channel // reduction, 3, padding=1, bias=False)
        self.idct_conv = nn.Conv1d(channel // reduction, channel, 1, bias=False)
        
        # LIF神经元参数
        self.threshold = nn.Parameter(torch.ones(channel) * 0.5)
        self.leak = nn.Parameter(torch.ones(channel) * 0.2)
        
        # 膜电位将在forward中动态创建
        
    def dct_transform(self, x):
        """DCT变换 - 基于1.py的频域处理思想"""
        # x: (B, C, L)
        B, C, L = x.shape
        
        # 使用FFT进行频域变换
        x_freq = torch.fft.rfft(x, dim=-1)
        x_freq_real = x_freq.real
        
        # 保留低频成分
        if x_freq_real.size(-1) > self.freq_num:
            x_freq_real = x_freq_real[..., :self.freq_num]
            
        return x_freq_real
    
    def idct_transform(self, x_freq, target_length):
        """逆DCT变换"""
        # 补零到目标长度
        if x_freq.size(-1) < target_length // 2 + 1:
            pad_size = target_length // 2 + 1 - x_freq.size(-1)
            x_freq = F.pad(x_freq, (0, pad_size))
        
        # 转换为复数并逆变换
        x_freq_complex = torch.complex(x_freq, torch.zeros_like(x_freq))
        x_reconstructed = torch.fft.irfft(x_freq_complex, n=target_length, dim=-1)
        
        return x_reconstructed

    def forward(self, x):
        # x: (step, B, C, L) or (B, C, L)
        if len(x.shape) == 3:
            x = x.unsqueeze(0).repeat(self.step, 1, 1, 1)
            
        step, B, C, L = x.shape
        
        # 每次前向传播重新初始化膜电位
        membrane = torch.zeros(B, C, L, device=x.device)
        
        outputs = []
        for t in range(step):
            current_input = x[t]  # (B, C, L)
            
            # DCT频域处理
            x_freq = self.dct_transform(current_input)  # (B, C, freq_num)
            
            # 频域卷积处理
            x_freq = self.dct_conv(x_freq)  # (B, C//reduction, freq_num)
            x_freq = F.relu(x_freq)
            x_freq = self.freq_conv(x_freq)  # (B, C//reduction, freq_num)
            x_freq = F.relu(x_freq)
            x_freq = self.idct_conv(x_freq)  # (B, C, freq_num)
            
            # 逆DCT变换
            enhanced_input = self.idct_transform(x_freq, L)  # (B, C, L)
            
            # LIF神经元动力学
            membrane = self.leak.view(1, -1, 1) * membrane + enhanced_input
            
            # 脉冲发放
            spike_mask = (membrane > self.threshold.view(1, -1, 1)).float()
            output = spike_mask * membrane
            
            # 重置膜电位
            membrane = membrane * (1 - spike_mask)
            
            outputs.append(output)
        
        return torch.stack(outputs, dim=0)  # (step, B, C, L)

class DctSpikeAttention(nn.Module):
    """
    DCT Spike Attention mechanism combining DCT processing with attention
    """
    def __init__(self, d_model, num_heads=8, step=2, freq_num=32, reduction=16):
        super(DctSpikeAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.step = step
        
        # DCT Spatial LIF处理
        self.dct_lif = DctSpatialLIF1D(step=step, channel=d_model, freq_num=freq_num, reduction=reduction)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.1)
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        residual = x
        
        # 转换为DCT处理格式 (B, C, L)
        x_dct = x.permute(0, 2, 1)
        
        # DCT Spike处理
        dct_out = self.dct_lif(x_dct)  # (step, B, C, L)
        
        # 时间平均并转回 (B, L, C)
        dct_out = dct_out.mean(dim=0).permute(0, 2, 1)
        
        # 多头注意力
        attn_out, _ = self.attention(x, x, x)
        
        # 特征融合
        combined = torch.cat([dct_out, attn_out], dim=-1)
        fused = self.fusion(combined)
        
        # 残差连接和归一化
        output = self.norm1(fused + residual)
        
        return output

class DctEnhancedEncoder(nn.Module):
    """
    DCT增强编码器，融合DCT注意力和标准Transformer
    """
    def __init__(self, configs):
        super(DctEnhancedEncoder, self).__init__()
        
        # DCT Spike注意力
        self.dct_attention = DctSpikeAttention(
            configs.d_model, 
            num_heads=configs.n_heads,
            step=4,  # 脉冲步数
            freq_num=48,
            reduction=16
        )
        
        # 标准Transformer编码器
        self.transformer_encoder = Encoder(
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
        
        # 自适应融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
        self.final_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x, attn_mask=None):
        # DCT增强处理
        dct_out = self.dct_attention(x)
        
        # Transformer处理
        transformer_out, attns = self.transformer_encoder(x, attn_mask=attn_mask)
        
        # 自适应融合
        weight = torch.sigmoid(self.fusion_weight)
        output = weight * dct_out + (1 - weight) * transformer_out
        
        # 最终处理
        output = self.final_norm(output)
        output = self.dropout(output)
        
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
    ZAAB: AdaWaveNet with DCT Spatial LIF attention from 1.py
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

        # DCT增强编码器（替代标准编码器）
        self.encoder = DctEnhancedEncoder(configs)
        
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
        
        # DCT增强编码器处理
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
