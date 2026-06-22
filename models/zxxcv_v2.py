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

class ImprovedChannelAttention(nn.Module):
    """
    改进的通道注意力 - 基于实验结果优化
    """
    def __init__(self, channels, reduction=8):  # 减少reduction提升表达能力
        super(ImprovedChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 增加网络深度和非线性
        mid_channels = max(16, channels // reduction)  # 提高最小通道数
        self.CA_fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # 添加dropout防止过拟合
            nn.Linear(mid_channels, mid_channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels // 2, channels, bias=False),
            nn.Sigmoid(),
        )
        
        # 添加残差连接的权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        # x: (B, C, L)
        b, c, l = x.size()
        
        # 双路池化获取更丰富的全局信息
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        # 分别计算注意力权重
        avg_weight = self.CA_fc(avg_out).view(b, c, 1)
        max_weight = self.CA_fc(max_out).view(b, c, 1)
        
        # 自适应融合两种池化的权重
        channel_weight = (avg_weight + max_weight) / 2
        
        # 添加残差连接
        residual_factor = torch.sigmoid(self.residual_weight)
        return x * (1 + channel_weight * residual_factor)

class EnhancedTemporalAttention(nn.Module):
    """
    增强的时序注意力 - 多尺度+位置编码
    """
    def __init__(self, channels):
        super(EnhancedTemporalAttention, self).__init__()
        
        # 多尺度卷积组
        self.multi_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=k, padding=k//2, groups=channels//4),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True)
            ) for k in [3, 5, 7]  # 多种卷积核大小
        ])
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv1d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 位置注意力
        self.pos_attention = nn.Sequential(
            nn.Conv1d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (B, C, L)
        identity = x
        
        # 多尺度特征提取
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            multi_scale_features.append(conv(x))
        
        # 融合多尺度特征
        fused = torch.cat(multi_scale_features, dim=1)
        fused = self.fusion(fused)
        
        # 位置注意力
        pos_weight = self.pos_attention(fused)
        
        return identity + fused * pos_weight

class AdaptiveSpatialAttention(nn.Module):
    """
    自适应空间注意力 - 动态膨胀率
    """
    def __init__(self, in_ch, out_ch):
        super(AdaptiveSpatialAttention, self).__init__()
        
        # 自适应膨胀率 - 根据序列长度动态调整
        self.base_rates = [1, 2, 4, 8, 16]  # 更密集的膨胀率
        
        self.SA_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=rate, dilation=rate),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(0.05)  # 轻微dropout
            ) for rate in self.base_rates
        ])
        
        # 注意力权重学习
        self.attention_weights = nn.Parameter(torch.ones(len(self.base_rates)) / len(self.base_rates))
        
        # 输出融合
        self.SA_out_conv = nn.Sequential(
            nn.Conv1d(len(self.base_rates) * out_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # x: (B, C, L)
        sa_outs = []
        weights = F.softmax(self.attention_weights, dim=0)
        
        for i, block in enumerate(self.SA_blocks):
            sa_out = block(x) * weights[i]  # 加权输出
            sa_outs.append(sa_out)
        
        sa_out = torch.cat(sa_outs, dim=1)
        sa_out = self.SA_out_conv(sa_out)
        return sa_out

class OptimizedSynergeticAttention(nn.Module):
    """
    优化的协同注意力 - 基于实验结果改进
    """
    def __init__(self, channels):
        super(OptimizedSynergeticAttention, self).__init__()
        self.channels = channels
        
        # 三种改进的注意力机制
        self.channel_attention = ImprovedChannelAttention(channels, reduction=8)
        self.temporal_attention = EnhancedTemporalAttention(channels)
        self.spatial_attention = AdaptiveSpatialAttention(channels, channels)
        
        # 动态权重学习
        self.dynamic_weights = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, 3, 1),
            nn.Softmax(dim=1)
        )
        
        # 输出处理
        self.output_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.final_norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        # x: (B, C, L)
        residual = x
        b, c, l = x.shape
        
        # 三种注意力机制
        ca_out = self.channel_attention(x)
        ta_out = self.temporal_attention(x)
        sa_out = self.spatial_attention(x)
        
        # 动态权重计算
        weights = self.dynamic_weights(x)  # (B, 3, 1)
        
        # 协同融合
        synergistic_out = (weights[:, 0:1, :] * ca_out + 
                          weights[:, 1:2, :] * ta_out + 
                          weights[:, 2:3, :] * sa_out)
        
        # 输出处理
        out = self.output_conv(synergistic_out)
        
        # 残差连接 + LayerNorm
        out = out + residual
        out = out.transpose(1, 2)  # (B, L, C)
        out = self.final_norm(out)
        out = out.transpose(1, 2)  # (B, C, L)
        
        return out

class ImprovedMLP(nn.Module):
    """
    改进的MLP - 更好的局部建模
    """
    def __init__(self, feature_size, forward_expansion=4, dropout=0.1):  # 增加expansion
        super(ImprovedMLP, self).__init__()
        
        expanded_size = forward_expansion * feature_size
        
        # 第一层线性变换
        self.linear1 = nn.Linear(feature_size, expanded_size)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        # 深度卷积 - 增强局部建模
        self.depthwise_conv = nn.Conv1d(
            expanded_size, expanded_size, 
            kernel_size=5, padding=2, groups=expanded_size  # 增大卷积核
        )
        
        # 逐点卷积
        self.pointwise_conv = nn.Conv1d(expanded_size, expanded_size, 1)
        
        # 中间归一化和激活
        self.norm = nn.BatchNorm1d(expanded_size)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        
        # 输出层
        self.linear2 = nn.Linear(expanded_size, feature_size)
        
    def forward(self, x):
        # x: (B, L, C)
        b, l, c = x.size()
        
        # 线性变换
        x = self.linear1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        
        # 转换为卷积格式
        x = x.transpose(1, 2)  # (B, C, L)
        
        # 卷积处理
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.norm(x)
        x = self.act2(x)
        x = self.dropout2(x)
        
        # 转换回序列格式
        x = x.transpose(1, 2)  # (B, L, C)
        
        # 输出线性层
        out = self.linear2(x)
        
        return out

class OptimizedSMABlock(nn.Module):
    """
    优化的SMA块 - 基于实验结果改进
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1, forward_expansion=4):
        super(OptimizedSMABlock, self).__init__()
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 多头自注意力 - 增加头数
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # 优化的协同注意力
        self.synergistic_attention = OptimizedSynergeticAttention(d_model)
        
        # 改进的MLP
        self.improved_mlp = ImprovedMLP(d_model, forward_expansion, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        # 残差权重
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # x: (B, L, C)
        
        # 第一个残差块：自注意力
        residual1 = x
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(residual1 + self.dropout(attn_out) * self.alpha)
        
        # 第二个残差块：协同注意力
        residual2 = x
        x_conv = x.transpose(1, 2)  # (B, C, L)
        sma_out = self.synergistic_attention(x_conv)
        sma_out = sma_out.transpose(1, 2)  # (B, L, C)
        x = self.norm2(residual2 + sma_out * self.beta)
        
        # 第三个残差块：MLP
        residual3 = x
        mlp_out = self.improved_mlp(x)
        x = self.norm3(residual3 + self.dropout(mlp_out))
        
        return x

class SuperiorEncoder(nn.Module):
    """
    卓越编码器 - 集成所有改进
    """
    def __init__(self, configs):
        super(SuperiorEncoder, self).__init__()
        
        # 多个优化的SMA块
        self.sma_blocks = nn.ModuleList([
            OptimizedSMABlock(configs.d_model, configs.n_heads, configs.dropout, forward_expansion=4)
            for _ in range(3)  # 增加到3个块
        ])
        
        # 轻量级Transformer编码器
        self.transformer_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), 
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff // 2,  # 减少FFN维度平衡计算
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(max(1, configs.e_layers - 1))  # 减少Transformer层数
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # 改进的特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model * 2),
            nn.LayerNorm(configs.d_model * 2),
            nn.GELU(),
            nn.Dropout(configs.dropout * 0.3),
            nn.Linear(configs.d_model * 2, configs.d_model),
        )
        
        self.norm = nn.LayerNorm(configs.d_model)
        
    def forward(self, x, attn_mask=None):
        # x: (B, L, C)
        identity = x
        
        # SMA处理
        sma_out = x
        for sma_block in self.sma_blocks:
            sma_out = sma_block(sma_out)
        
        # Transformer处理
        transformer_out, attns = self.transformer_encoder(x, attn_mask=attn_mask)
        
        # 特征融合
        fused_features = torch.cat([sma_out, transformer_out], dim=-1)
        output = self.feature_fusion(fused_features)
        
        # 残差连接和归一化
        output = self.norm(identity + output)
        
        return output, attns

# 复用原有的基础模块
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
    ZXXCV_V2: 基于实验结果优化的AdaWaveNet + SMAFormer模型
    
    主要改进:
    1. 更深的注意力网络 - 提升表达能力
    2. 动态权重学习 - 自适应特征融合
    3. 改进的MLP结构 - 更好的局部建模
    4. 优化的残差连接 - 稳定训练过程
    5. 增强的正则化 - 防止过拟合
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

        # 卓越编码器
        self.encoder = SuperiorEncoder(configs)
        
        # 改进的预测头
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Sequential(
                nn.Linear(self.seq_len, configs.pred_len * 3),  # 增加中间维度
                nn.LayerNorm(configs.pred_len * 3),
                nn.GELU(),
                nn.Dropout(configs.dropout * 0.3),
                nn.Linear(configs.pred_len * 3, configs.pred_len * 2),
                nn.LayerNorm(configs.pred_len * 2),
                nn.GELU(),
                nn.Dropout(configs.dropout * 0.2),
                nn.Linear(configs.pred_len * 2, configs.pred_len),
            )
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
        
        # 卓越编码器处理
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
        # 与forecast相同的实现...
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, self.clusters)

    def anomaly_detection(self, x_enc):
        # 与forecast相同的实现...
        return self.forecast(x_enc, None, None, None, self.clusters)

    def super_resolution(self, x_enc):
        # 与forecast相同的实现...
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



















