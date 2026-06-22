import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from layers.LiftingScheme import LiftingScheme, InverseLiftingScheme
from layers.Invertible import RevIN

def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)

# ========== 简化优化模块 ==========

class SimpleChannelMixer(nn.Module):
    """简化的通道混合器"""
    def __init__(self, channels, dropout=0.05):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, C, L)
        out = self.conv(x)
        out = self.norm(out)
        out = self.dropout(out)
        return x + 0.1 * out  # 轻量级残差连接

class OptimizedWaveletBlock(nn.Module):
    """优化的小波处理块"""
    def __init__(self, channels, dropout=0.05):
        super().__init__()
        # 简化的小波特征增强
        self.pre_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.post_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x: (B, C, L)
        residual = x
        
        # 预处理
        out = self.pre_conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # 后处理
        out = self.post_conv(out)
        out = self.norm(out)
        
        # 残差连接
        return self.activation(residual + out)

class SimplifiedMultiBranchProcessor(nn.Module):
    """简化的多分支处理模块 - 减少复杂度"""
    def __init__(self, d_model, num_branches=2, dropout=0.05):
        super().__init__()
        self.num_branches = num_branches
        self.d_model = d_model
        
        # 简化的分支处理器
        self.branch1 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        self.branch2 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 简化权重
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, L, D)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        
        # 简单加权融合
        alpha = torch.sigmoid(self.alpha)
        weighted_output = alpha * out1 + (1 - alpha) * out2
        weighted_output = self.dropout(weighted_output)
        
        return self.norm(x + 0.1 * weighted_output)  # 减小残差权重

class LightweightSelfGating(nn.Module):
    """轻量级自身门控机制"""
    def __init__(self, d_model, dropout=0.05):
        super().__init__()
        # 简化门控网络
        self.gate_linear = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, L, D) or (B, C, L)
        gate = self.gate_linear(x)
        # 减弱门控影响，避免过度抑制
        gate = 0.5 + 0.5 * gate  # 将门控值范围从[0,1]调整到[0.5,1]
        gated_output = x * gate
        gated_output = self.dropout(gated_output)
        return self.norm(x + 0.1 * (gated_output - x))  # 减小门控影响

class CrossScaleAttention(nn.Module):
    """跨尺度注意力机制（参考PatchTST + iTransformer）"""
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        residual = x
        
        # Multi-head attention
        q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        
        output = self.w_o(attn_output)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.norm(residual + output)
        
        return output

class DecomposableMixing(nn.Module):
    """可分解混合模块（参考TimeMixer）"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 季节分支
        self.season_mixer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        # 趋势分支
        self.trend_mixer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (B, L, D) 原始输入
        # 简化为只处理原始输入，不依赖外部分解
        out = self.season_mixer(x) + self.trend_mixer(x)
        return self.norm(x + out)

class ECAAttention(nn.Module):
    """ECA-Net: 高效通道注意力机制（轻量级）"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (B, C, L)
        y = self.gap(x)  # (B, C, 1)
        y = y.permute(0, 2, 1)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1)  # (B, C, 1)
        return x * y

class ChannelMixer(nn.Module):
    """通道混合模块（参考DUET + TimeMixer）"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.mixer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        # 添加LayerNorm（参考TimeMixer）
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (B, L, D)
        return self.norm(x + self.mixer(x))

class MultiScaleFeatureExtraction(nn.Module):
    """轻量级多尺度特征提取模块"""
    def __init__(self, in_channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        
        # 使用ECA注意力
        self.eca = ECAAttention(in_channels)
        
        # 融合
        self.fusion = nn.Conv1d(in_channels * 2, in_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, C, L)
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        
        # 多尺度融合
        multi_scale = torch.cat([y1, y2], dim=1)
        x_att = self.fusion(multi_scale)
        
        # 应用ECA注意力
        x_att = self.eca(x_att)
        
        # 残差连接 + dropout
        x_att = self.dropout(x_att)
        return x_att + x

class EfficientAdditiveAttention(nn.Module):
    """高效加法注意力机制"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.scale_factor = d_model ** -0.5
        
        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(d_model, d_model)
        self.w_a = nn.Parameter(torch.randn(d_model, 1))
        self.proj = nn.Linear(d_model, d_model)
        self.final = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        
        # 生成query和key
        query = self.to_query(x)
        key = self.to_key(x)
        
        # 标准化
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        
        # 学习注意力权重
        query_weight = query @ self.w_a
        A = query_weight * self.scale_factor
        A = F.normalize(A, dim=1)
        
        # 加权
        q = torch.sum(A * query, dim=1).reshape(B, 1, -1)
        out = self.proj(q * key) + query
        out = self.final(out)
        out = self.dropout(out)
        
        return out

class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series"""
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
    """Series decomposition block"""
    def __init__(self, kernel_size=24, stride=1, imputation=False):
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
        output = torch.cat(output, dim=1)
        return output

class Model(nn.Module):
    """YANGNet2 - 增强版YANGNet模型"""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'super_resolution':
            self.seq_len = self.seq_len // configs.sr_ratio
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.n_clusters = configs.n_clusters
        
        # 基础组件
        self.series_decomp = series_decomp(imputation = self.task_name=='imputation')
        self.rev_seasonal = RevIN(configs.enc_in)
        self.rev_trend = RevIN(configs.enc_in)
        self.trend_linear = ClusteredLinear(configs.n_clusters, configs.enc_in, self.seq_len, configs.pred_len)
        
        # ========== 简化优化模块 ==========
        self.simple_channel_mixer = SimpleChannelMixer(configs.enc_in, configs.dropout)
        self.optimized_wavelet_block = OptimizedWaveletBlock(configs.enc_in, configs.dropout)
        self.cross_scale_attention = CrossScaleAttention(configs.d_model, configs.n_heads, configs.dropout)
        self.decomposable_mixing = DecomposableMixing(configs.d_model, configs.d_ff, configs.dropout)
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len // (2 ** configs.lifting_levels), configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # 小波变换层级
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
                    nn.Dropout(configs.dropout)
                )
            )
            self.coef_linear_levels.add_module(
                'linear_level_'+str(i),
                nn.Sequential(
                    nn.Linear(input_size, input_size * expand_ratio),
                    nn.Dropout(configs.dropout)
                )
            )
            self.coef_dec_levels.add_module(
                'linear_level_'+str(i),
                nn.Sequential(
                    nn.Linear(input_size, input_size * expand_ratio),
                    nn.Dropout(configs.dropout)
                )
            )

        self.input_size = input_size
        
        # 解码器层级
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
        
        # 投影层
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(self.seq_len, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(self.seq_len, self.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(self.seq_len, self.seq_len, bias=True)
        if self.task_name == 'super_resolution':
            self.projection = nn.Linear(configs.pred_len, configs.pred_len, bias=True)

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

        self.register_buffer('clusters', None)
    
    def _simple_kmeans(self, features, n_clusters, max_iters=10):
        """简单的K-means聚类实现"""
        n_samples, n_features = features.shape
        device = features.device
        
        n_clusters = min(n_clusters, n_samples)
        centroids = features[torch.randperm(n_samples)[:n_clusters]]
        
        for _ in range(max_iters):
            distances = torch.cdist(features, centroids)
            clusters = torch.argmin(distances, dim=1)
            
            new_centroids = torch.zeros_like(centroids)
            for i in range(n_clusters):
                mask = clusters == i
                if mask.sum() > 0:
                    new_centroids[i] = features[mask].mean(dim=0)
                else:
                    if i < n_samples:
                        new_centroids[i] = features[torch.randint(0, n_samples, (1,), device=device)]
                    else:
                        new_centroids[i] = centroids[i]
            
            if torch.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids
        
        return clusters

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, clusters):
        # 使用双重分解：移动平均 + 频率域分解
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        
        # Normalization
        x_enc = x.permute(0,2,1)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, _, N = x_enc.shape

        moving_mean = self.rev_trend(moving_mean, 'norm')
        x_enc = x_enc.permute(0,2,1)
        
        # ========== 简化的通道混合和小波优化 ==========
        x_enc_mixed = self.simple_channel_mixer(x_enc)
        
        # ========== 优化的小波预处理 ==========
        x_enc_enhanced = self.optimized_wavelet_block(x_enc_mixed)
        
        encoded_coefficients = []
        x_embedding_levels = []
        coef_embedding_levels = []
        
        # Encoding
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc_enhanced, r, details = l(x_enc_enhanced)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc_enhanced))
        
        # Embedding
        x_enc_enhanced = x_enc_enhanced.permute(0,2,1)
        enc_out = self.enc_embedding(x_enc_enhanced, None)
        
        # ========== 跨尺度注意力机制 ==========
        enc_out_attention = self.cross_scale_attention(enc_out)
        
        # ========== 可分解混合（季节+趋势分支） ==========
        enc_out_enhanced = self.decomposable_mixing(enc_out_attention)
        
        enc_out, attns = self.encoder(enc_out_enhanced, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        
        # Decoding
        for dec, x_emb_level, coef_emb_level, c_linear in zip(self.decoder_levels, x_embedding_levels[::-1], coef_embedding_levels[::-1], self.coef_dec_levels[::-1]):
            details = encoded_coefficients.pop()
            details = coef_emb_level + c_linear(details)
            x_dec = x_dec + x_emb_level
            x_dec = dec(x_dec, details)
        
        # 投影
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        
        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        moving_mean_out = self.trend_linear(moving_mean.permute(0,2,1), self.clusters).permute(0,2,1)
        moving_mean_out = self.rev_trend(moving_mean_out, 'denorm')
        
        # 简单融合（保持与原始YANGNet一致）
        dec_out = dec_out + moving_mean_out
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, C = x_enc.shape
        x_cluster = x_enc.permute(2,0,1).view(C, B * L)
        if self.clusters is None:
            x_var = torch.var(x_cluster, dim=1)
            x_mean = torch.mean(x_cluster, dim=1)
            features = torch.stack([x_mean, x_var], dim=1)
            features_norm = F.normalize(features, p=2, dim=1)
            clusters = self._simple_kmeans(features_norm, self.n_clusters)
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
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        if self.task_name == 'super_resolution':
            dec_out = self.super_resolution(x_enc)
            return dec_out
        return None

