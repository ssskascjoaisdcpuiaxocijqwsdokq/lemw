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

class EnhancedTemporalGlobal(nn.Module):
    """
    增强时序全局特征提取 - 多尺度全局建模
    """
    def __init__(self, dim):
        super().__init__()
        # 多尺度全局池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 全局上下文建模
        self.global_context = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (B, C, L)
        avg_feat = self.avg_pool(x).squeeze(-1)  # (B, C)
        max_feat = self.max_pool(x).squeeze(-1)  # (B, C)
        
        # 融合全局特征
        global_feat = torch.cat([avg_feat, max_feat], dim=1)  # (B, 2C)
        global_weight = self.global_context(global_feat)  # (B, C)
        
        return global_weight.unsqueeze(-1)  # (B, C, 1)

class MultiScaleTemporalLocal(nn.Module):
    """
    多尺度时序局部特征提取 - 捕获不同时间尺度的模式
    """
    def __init__(self, dim):
        super().__init__()
        # 确保输出通道数能被整除
        out_channels = max(1, dim // 4)  # 每个尺度的输出通道数
        
        # 多尺度卷积分支 - 修复分组卷积问题
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, out_channels, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ) for k in [3, 5, 7]  # 不同的时间尺度
        ])
        
        # 额外的深度卷积分支
        self.depth_conv = nn.Sequential(
            nn.Conv1d(dim, out_channels, kernel_size=3, padding=1, groups=min(dim, out_channels)),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合 - 4个分支的输出
        total_channels = out_channels * 4
        self.fusion = nn.Sequential(
            nn.Conv1d(total_channels, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )
        
        # 时序位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, dim, 1) * 0.02)
    
    def forward(self, x):
        # x: (B, C, L)
        # 多尺度特征提取
        scale_features = []
        for scale_conv in self.scales:
            scale_features.append(scale_conv(x))
        
        # 深度卷积特征
        depth_feat = self.depth_conv(x)
        scale_features.append(depth_feat)
        
        # 拼接多尺度特征
        multi_scale = torch.cat(scale_features, dim=1)  # (B, total_channels, L)
        
        # 特征融合
        fused = self.fusion(multi_scale)
        
        # 添加位置编码
        fused = fused + self.pos_encoding.expand_as(fused)
        
        return fused

class AdaptiveTemporalFusion(nn.Module):
    """
    自适应时序融合 - 智能的局部-全局特征融合
    """
    def __init__(self, dim):
        super().__init__()
        self.local = MultiScaleTemporalLocal(dim)
        self.global_ = EnhancedTemporalGlobal(dim)
        
        # 自适应融合权重网络
        self.fusion_net = nn.Sequential(
            nn.Conv1d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim),
            nn.Sigmoid()
        )
        
        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.2))
        
    def forward(self, x):
        # x: (B, C, L)
        identity = x
        
        # 局部和全局特征提取
        local_feat = self.local(x)  # (B, C, L)
        global_weight = self.global_(x)  # (B, C, 1)
        
        # 全局权重调制局部特征
        modulated_local = local_feat * global_weight.expand_as(local_feat)
        
        # 自适应融合
        concat_feat = torch.cat([local_feat, modulated_local], dim=1)  # (B, 2C, L)
        fusion_weight = self.fusion_net(concat_feat)  # (B, C, L)
        
        # 加权融合
        fused = fusion_weight * modulated_local + (1 - fusion_weight) * local_feat
        
        # 残差连接
        residual_factor = torch.sigmoid(self.residual_weight)
        output = fused + identity * residual_factor
        
        return output

class EnhancedTemporalMASAG(nn.Module):
    """
    增强时序MASAG - 更好的时序数据建模能力
    
    核心改进：
    1. 多尺度特征提取 - 捕获不同时间尺度的模式
    2. 自适应特征融合 - 智能权重分配
    3. 增强注意力机制 - 更精确的时序关注
    4. 优化残差连接 - 更好的梯度流动
    """
    def __init__(self, dim):
        super().__init__()
        # 增强的特征融合
        self.fusion = AdaptiveTemporalFusion(dim)
        
        # 多头时序注意力
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=8, 
            dropout=0.1,
            batch_first=False  # (L, B, C)
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // 8, dim, 1),
            nn.Sigmoid()
        )
        
        # 输出处理
        self.output_proj = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # 自适应权重
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.beta = nn.Parameter(torch.tensor(0.7))
        
    def forward(self, x):
        # x: (B, C, L)
        identity = x
        B, C, L = x.shape
        
        # 1. 自适应特征融合
        fused = self.fusion(x)  # (B, C, L)
        
        # 2. 时序注意力 (需要转换维度)
        x_seq = fused.permute(2, 0, 1)  # (L, B, C)
        attn_out, _ = self.temporal_attention(x_seq, x_seq, x_seq)
        attn_out = attn_out.permute(1, 2, 0)  # (B, C, L)
        
        # 第一个残差连接
        alpha = torch.sigmoid(self.alpha)
        x1 = alpha * attn_out + (1 - alpha) * fused
        x1 = x1.permute(0, 2, 1)  # (B, L, C)
        x1 = self.norm1(x1)
        x1 = x1.permute(0, 2, 1)  # (B, C, L)
        
        # 3. 通道注意力
        ch_weight = self.channel_attention(x1)  # (B, C, 1)
        x2 = x1 * ch_weight
        
        # 4. 输出投影
        output = self.output_proj(x2)
        
        # 第二个残差连接
        beta = torch.sigmoid(self.beta)
        output = beta * output + (1 - beta) * identity
        
        # 最终归一化
        output = output.permute(0, 2, 1)  # (B, L, C)
        output = self.norm2(output)
        output = output.permute(0, 2, 1)  # (B, C, L)
        
        return output

class EnhancedMASAGEncoder(nn.Module):
    """
    增强MASAG编码器 - 更强的时序建模能力
    """
    def __init__(self, configs):
        super(EnhancedMASAGEncoder, self).__init__()
        
        # 增强的MASAG块
        self.masag_block = EnhancedTemporalMASAG(configs.d_model)
        
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
        
        # 智能特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model * 2),
            nn.LayerNorm(configs.d_model * 2),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.LayerNorm(configs.d_model)
        )
        
        # 动态权重网络
        self.weight_net = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(configs.d_model // 4, 2),
            nn.Softmax(dim=-1)
        )
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(configs.d_model)
        
    def forward(self, x, attn_mask=None):
        # x: (B, L, C)
        identity = x
        B, L, C = x.shape
        
        # 转换为卷积格式进行MASAG处理
        x_conv = x.transpose(1, 2)  # (B, C, L)
        
        # 增强的MASAG处理
        masag_out = self.masag_block(x_conv)  # (B, C, L)
        masag_out = masag_out.transpose(1, 2)  # (B, L, C)
        
        # Transformer处理
        transformer_out, attns = self.transformer_encoder(x, attn_mask=attn_mask)
        
        # 动态权重计算
        # 使用全局平均池化获取序列级特征
        global_feat = torch.mean(x, dim=1)  # (B, C)
        weights = self.weight_net(global_feat)  # (B, 2)
        
        # 动态加权融合
        weighted_masag = masag_out * weights[:, 0:1].unsqueeze(-1)  # (B, L, C)
        weighted_transformer = transformer_out * weights[:, 1:2].unsqueeze(-1)  # (B, L, C)
        
        # 特征拼接和融合
        concat_features = torch.cat([weighted_masag, weighted_transformer], dim=-1)  # (B, L, 2C)
        fused_output = self.feature_fusion(concat_features)  # (B, L, C)
        
        # 残差连接和输出归一化
        output = self.output_norm(fused_output + identity)
        
        return output, attns

# 复用AdaWaveNet的基础模块
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
    ZANNN: AdaWaveNet + 增强时序MASAG注意力 (V2)
    
    主要创新:
    1. 多尺度时序特征提取 - 捕获不同时间尺度的模式
    2. 自适应特征融合 - 智能权重分配和动态融合
    3. 增强注意力机制 - 时序注意力 + 通道注意力双重增强
    4. 动态权重网络 - 根据输入自适应调整融合策略
    
    核心改进:
    - 多尺度卷积捕获不同时间尺度特征
    - 全局-局部特征智能融合
    - 多头时序注意力增强序列建模
    - 动态权重网络实现自适应特征融合
    - 优化的残差连接和归一化策略
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

        # 增强MASAG编码器
        self.encoder = EnhancedMASAGEncoder(configs)
        
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
        
        # 增强MASAG编码器处理
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
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, self.clusters)

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
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == 'super_resolution':
            dec_out = self.super_resolution(x_enc)
            return dec_out
        return None
