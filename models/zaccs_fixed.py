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

class OptimizedHWD1D(nn.Module):
    """
    优化的1D Haar小波下采样模块 - 修复性能问题
    """
    def __init__(self, in_ch, out_ch):
        super(OptimizedHWD1D, self).__init__()
        
        # 问题修复1: 减少通道扩展，避免信息瓶颈
        self.conv_process = nn.Sequential(
            nn.Conv1d(in_ch * 2, in_ch, kernel_size=1, stride=1),  # 先降维
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),  # 使用ReLU替代GELU，减少计算复杂度
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1),    # 再映射到输出维度
        )
        
    def haar_wavelet_1d(self, x):
        """
        优化的1D Haar小波变换 - 修复数值稳定性问题
        """
        # x: (B, C, L)
        B, C, L = x.shape
        
        # 问题修复2: 改进填充策略，避免边界效应
        if L % 2 != 0:
            # 使用复制填充而不是反射填充，保持数值稳定性
            x = F.pad(x, (0, 1), mode='replicate')
            L = L + 1
        
        # 问题修复3: 改进小波变换，避免除法导致的数值不稳定
        # 低频分量 (近似系数) - 使用加权平均
        low_freq = (x[:, :, ::2] + x[:, :, 1::2]) * 0.7071  # sqrt(2)/2，保持能量
        
        # 高频分量 (细节系数) - 使用加权差分
        high_freq = (x[:, :, ::2] - x[:, :, 1::2]) * 0.7071
        
        return low_freq, high_freq
    
    def forward(self, x):
        # x: (B, C, L)
        low_freq, high_freq = self.haar_wavelet_1d(x)
        
        # 将低频和高频分量在通道维度拼接
        x_combined = torch.cat([low_freq, high_freq], dim=1)  # (B, 2*C, L//2)
        
        # 特征处理
        x_out = self.conv_process(x_combined)  # (B, out_ch, L//2)
        
        return x_out

class LightweightHWDEncoder(nn.Module):
    """
    轻量级HWD编码器 - 专注于性能优化
    """
    def __init__(self, configs):
        super(LightweightHWDEncoder, self).__init__()
        
        # 问题修复4: 减少HWD层数，避免过度下采样导致信息丢失
        self.hwd_layer = OptimizedHWD1D(configs.d_model, configs.d_model)
        
        # 问题修复5: 简化特征增强，避免过度复杂化
        self.feature_enhance = nn.Sequential(
            nn.Conv1d(configs.d_model, configs.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(configs.d_model),
            nn.ReLU(inplace=True),
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
        
        # 问题修复6: 简化融合机制，使用学习权重而非复杂门控
        self.fusion_weight = nn.Parameter(torch.tensor(0.3))  # HWD权重
        self.final_norm = nn.LayerNorm(configs.d_model)
        
    def forward(self, x, attn_mask=None):
        # x: (B, L, C)
        B, L, C = x.shape
        
        # HWD处理
        x_hwd = x.permute(0, 2, 1)  # (B, C, L)
        hwd_out = self.hwd_layer(x_hwd)  # (B, C, L//2)
        
        # 特征增强
        hwd_out = self.feature_enhance(hwd_out)
        
        # 问题修复7: 改进上采样策略，使用线性插值保持平滑性
        if hwd_out.size(-1) != L:
            hwd_out = F.interpolate(hwd_out, size=L, mode='linear', align_corners=True)
        
        hwd_out = hwd_out.permute(0, 2, 1)  # (B, L, C)
        
        # 标准Transformer处理
        transformer_out, attns = self.transformer_encoder(x, attn_mask=attn_mask)
        
        # 问题修复8: 简化融合，使用加权平均而非复杂门控
        weight = torch.sigmoid(self.fusion_weight)
        output = weight * hwd_out + (1 - weight) * transformer_out
        
        # 问题修复9: 添加残差连接，提升梯度流
        output = output + x
        
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
    ZACCS_Fixed: 修复性能问题的ZACCS模型
    主要修复:
    1. 数值稳定性问题
    2. 信息瓶颈问题  
    3. 过度复杂化问题
    4. 梯度流问题
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

        # 轻量级HWD编码器 (修复版本)
        self.encoder = LightweightHWDEncoder(configs)
        
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
        
        # 轻量级HWD编码器处理
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




































