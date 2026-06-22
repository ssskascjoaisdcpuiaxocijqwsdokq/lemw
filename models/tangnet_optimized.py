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

class LightweightSEAttention1D(nn.Module):
    """
    轻量级SE注意力机制 - 专注于降低MAE
    减少复杂度，提高预测精度
    """
    def __init__(self, channel, reduction=8):
        super().__init__()
        # 更保守的reduction，避免信息损失
        self.reduction = max(reduction, 4)  # 最小reduction=4
        hidden_dim = max(channel // self.reduction, 4)  # 最小4个神经元
        
        # 简化的SE网络
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channel, bias=False),
            nn.Sigmoid()  # 直接使用sigmoid，避免额外计算
        )
        
    def forward(self, x):
        """
        简化的SE注意力
        Args:
            x: 输入张量 (B, C, L)
        Returns:
            增强后的输出 (B, C, L)
        """
        B, C, L = x.size()
        
        # Squeeze: 全局平均池化
        y = self.avg_pool(x).view(B, C)
        
        # Excitation: 学习通道权重
        weight = self.fc(y).view(B, C, 1)
        
        # Scale: 应用注意力权重
        return x * weight

class OptimizedMovingAvg(nn.Module):
    """优化的移动平均 - 减少计算复杂度"""
    def __init__(self, kernel_size, stride):
        super(OptimizedMovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 简化的padding策略
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, self.kernel_size // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x

class OptimizedSeriesDecomp(nn.Module):
    """优化的序列分解 - 移除复杂的SE融合"""
    def __init__(self, kernel_size=24, stride=1, imputation=False):
        super(OptimizedSeriesDecomp, self).__init__()
        self.moving_avg = OptimizedMovingAvg(kernel_size, stride)

    def forward(self, x):
        # 简单直接的分解，避免SE引入的复杂性
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class OptimizedAdpWaveletBlock(nn.Module):
    """优化的小波块 - 选择性使用SE"""
    def __init__(self, configs, input_size):
        super(OptimizedAdpWaveletBlock, self).__init__()
        self.regu_details = configs.regu_details
        self.regu_approx = configs.regu_approx
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingScheme(configs.enc_in, k_size=configs.lifting_kernel_size, input_size=input_size)
        self.norm_x = normalization(configs.enc_in)
        self.norm_d = normalization(configs.enc_in)
        
        # 只在关键位置使用轻量级SE
        self.se_attention = LightweightSEAttention1D(
            channel=configs.enc_in,
            reduction=8
        )

    def forward(self, x):
        # 小波变换
        (c, d) = self.wavelet(x)
        
        # 轻量级SE增强，只增强近似系数
        c_enhanced = self.se_attention(c)
        x = c_enhanced

        # 正则化损失计算
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
    """逆小波变换块"""
    def __init__(self, configs, input_size):
        super(InverseAdpWaveletBlock, self).__init__()
        self.inverse_wavelet = InverseLiftingScheme(configs.enc_in, input_size=input_size, kernel_size=configs.lifting_kernel_size)

    def forward(self, c, d):
        reconstructed = self.inverse_wavelet(c, d)
        return reconstructed

class OptimizedClusteredLinear(nn.Module):
    """优化的聚类线性层 - 减少过拟合"""
    def __init__(self, n_clusters, enc_in, seq_len, pred_len):
        super().__init__()
        self.n_clusters = n_clusters
        self.enc_in = enc_in
        
        # 简化的线性层，减少过拟合
        self.linear_layers = nn.ModuleDict({
            str(cluster_id): nn.Sequential(
                nn.Linear(seq_len, pred_len),  # 直接映射，减少复杂度
                nn.Dropout(0.05)  # 轻微dropout
            ) for cluster_id in range(n_clusters)
        })
        
        # 轻量级SE用于特征选择
        self.se_attention = LightweightSEAttention1D(
            channel=enc_in,
            reduction=4  # 更保守的reduction
        )
        
    def forward(self, x, clusters):
        # 轻量级SE增强
        x_enhanced = self.se_attention(x)
        
        output = []
        assert self.enc_in == len(clusters)
        for channel in range(self.enc_in):
            cluster_id = str(clusters[channel].item())
            channel_data = x_enhanced[:, channel, :].unsqueeze(1)
            transformed_channel = self.linear_layers[cluster_id](channel_data)
            output.append(transformed_channel)
        output = torch.concat(output, dim=1)
        return output

class ResidualRefinement(nn.Module):
    """残差精炼模块 - 专注于减少预测误差"""
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        # 简单的残差网络
        self.refine_net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),  # 1x1卷积降低复杂度
        )
        
        # 残差权重
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        # 简单的残差连接
        refined = self.refine_net(x)
        return x + self.alpha * refined

class Model(nn.Module):
    """
    TangNet Optimized: 优化版本，专注于降低MAE
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
        
        # 优化的序列分解
        self.series_decomp = OptimizedSeriesDecomp(imputation = self.task_name=='imputation')
        self.rev_seasonal = RevIN(configs.enc_in)
        self.rev_trend = RevIN(configs.enc_in)
        
        # 优化的聚类线性层
        self.trend_linear = OptimizedClusteredLinear(configs.n_clusters, configs.enc_in, self.seq_len, configs.pred_len)
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len // (2 ** configs.lifting_levels), configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # 构建编码器层级
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
            # 优化的小波块
            self.encoder_levels.add_module(
                'encoder_level_'+str(i),
                OptimizedAdpWaveletBlock(configs, input_size)
            )
            in_planes *= 1
            input_size = input_size // 2 
            
            # 简化的线性层
            simple_linear = nn.Sequential(
                nn.Linear(input_size, input_size * expand_ratio),
                nn.Dropout(configs.dropout * 0.5)  # 减少dropout
            )
            
            self.linear_levels.add_module('linear_level_'+str(i), simple_linear)
            self.coef_linear_levels.add_module('coef_linear_level_'+str(i), simple_linear)
            self.coef_dec_levels.add_module('coef_dec_level_'+str(i), simple_linear)

        self.input_size = input_size
        
        # 构建解码器层级
        self.decoder_levels = nn.ModuleList()
        for i in range(configs.lifting_levels-1, -1, -1):
            self.decoder_levels.add_module(
                'decoder_level_'+str(i),
                InverseAdpWaveletBlock(configs, input_size=input_size)
            )
            in_planes //= 1
            input_size *= 2
        
        # 简化的低秩投影
        if self.task_name == "super_resolution":
            self.lowrank_projection = nn.Linear(configs.d_model, self.pred_len // (2 ** configs.lifting_levels), bias=True)
        else:
            self.lowrank_projection = nn.Linear(configs.d_model, self.seq_len // (2 ** configs.lifting_levels), bias=True)

        # Transformer编码器
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
        
        # 简化的投影层
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(self.seq_len, configs.pred_len, bias=True)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(self.seq_len, self.seq_len, bias=True)
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(self.seq_len, self.seq_len, bias=True)
        elif self.task_name == 'super_resolution':
            self.projection = nn.Linear(configs.pred_len, configs.pred_len, bias=True)

        # 残差精炼模块
        self.residual_refinement = ResidualRefinement(configs.enc_in, configs.dropout)

        self.register_buffer('clusters', None)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, clusters):
        # 优化的序列分解
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        
        # 归一化处理
        x_enc = x.permute(0,2,1)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, _, N = x_enc.shape

        # 趋势处理
        moving_mean = self.rev_trend(moving_mean, 'norm')
        x_enc = x_enc.permute(0,2,1)
        
        encoded_coefficients = []
        x_embedding_levels = []
        coef_embedding_levels = []
        
        # 优化的编码过程
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
        
        # Transformer编码
        x_enc = x_enc.permute(0,2,1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        
        # 解码过程
        for dec, x_emb_level, coef_emb_level, c_linear in zip(self.decoder_levels, x_embedding_levels[::-1], coef_embedding_levels[::-1], self.coef_dec_levels[::-1]):
            details = encoded_coefficients.pop()
            details = coef_emb_level + c_linear(details)
            x_dec = x_dec + x_emb_level
            x_dec = dec(x_dec, details)
            
        # 投影和反归一化
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        # 优化的趋势预测
        moving_mean_out = self.trend_linear(moving_mean.permute(0,2,1), self.clusters).permute(0,2,1)
        moving_mean_out = self.rev_trend(moving_mean_out, 'denorm')
        
        # 最终融合
        dec_out = dec_out + moving_mean_out
        
        # 残差精炼
        dec_out = self.residual_refinement(dec_out.permute(0,2,1)).permute(0,2,1)
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        base_output = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, self.clusters)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(base_output)
            refined_output = torch.where(mask_expanded.bool(), base_output, x_enc)
            return refined_output
        
        return base_output

    def anomaly_detection(self, x_enc):
        return self.forecast(x_enc, None, None, None, self.clusters)

    def super_resolution(self, x_enc):
        return self.forecast(x_enc, None, None, None, self.clusters)

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
        if self.task_name == 'super_resolution':
            dec_out = self.super_resolution(x_enc)
            return dec_out
        return None






