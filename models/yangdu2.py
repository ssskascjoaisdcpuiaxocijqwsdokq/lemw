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
from einops import rearrange

def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)

class EnhancedSEBlock(nn.Module):
    """增强SE注意力块 - 通道+空间注意力机制"""
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        mid_channels = max(1, channels // reduction)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mid_channels, channels, bias=False)
        )
        
        # 空间注意力
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel//2, bias=False),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        
        # 融合权重
        self.channel_weight = nn.Parameter(torch.tensor(0.7))
        self.spatial_weight = nn.Parameter(torch.tensor(0.3))
        
        # 残差缩放
        self.residual_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # x: [B, C, L]
        B, C, L = x.size()
        residual = x
        
        # 通道注意力
        avg_pool = self.avg_pool(x).view(B, C)
        max_pool = self.max_pool(x).view(B, C)
        
        avg_out = self.channel_fc(avg_pool)
        max_out = self.channel_fc(max_pool)
        channel_att = torch.sigmoid(avg_out + max_out).view(B, C, 1)
        
        # 空间注意力
        avg_spatial = torch.mean(x, dim=1, keepdim=True)  # [B, 1, L]
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, L]
        spatial_concat = torch.cat([avg_spatial, max_spatial], dim=1)  # [B, 2, L]
        spatial_att = self.spatial_conv(spatial_concat)  # [B, 1, L]
        
        # 加权融合
        channel_weight = torch.sigmoid(self.channel_weight)
        spatial_weight = torch.sigmoid(self.spatial_weight)
        
        enhanced_x = x * (channel_weight * channel_att + spatial_weight * spatial_att)
        
        # 残差连接
        return residual * self.residual_scale + enhanced_x

class EnhancedLinearExtractor(nn.Module):
    """
    增强的线性特征提取器 - 集成多尺度特征和注意力机制
    """
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.d_model = config.d_model
        self.enc_in = config.enc_in
        self.pred_len = config.pred_len
        
        # RevIN归一化
        self.revin = RevIN(config.enc_in)
        
        # 多尺度特征提取
        self.conv1d_layers = nn.ModuleList([
            nn.Conv1d(1, config.d_model // 4, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.d_model // 4 * 3, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # 增强SE注意力
        self.se_block = EnhancedSEBlock(config.d_model, reduction=max(4, config.d_model // 16))
        
        # 自适应池化
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1),
            nn.AdaptiveMaxPool1d(1),
            nn.AdaptiveAvgPool1d(config.seq_len // 4),
            nn.AdaptiveAvgPool1d(config.seq_len // 8)
        ])
        
        # 池化特征融合
        self.pool_fusion = nn.Sequential(
            nn.Linear(config.d_model * 4, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # 频域增强
        self.freq_enhance = nn.Sequential(
            nn.Linear(config.seq_len // 2 + 1, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.d_model)
        )
        
        # 重要性权重学习
        self.importance_layer = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, L, C] 或 [(B*C), L, 1]
        if x.dim() == 3 and x.shape[-1] == 1:
            # Channel Independent模式
            x = x.squeeze(-1)  # [(B*C), L]
            x = x.unsqueeze(1)  # [(B*C), 1, L]
            
            # 多尺度卷积特征提取
            conv_features = []
            for conv in self.conv1d_layers:
                feat = F.gelu(conv(x))  # [(B*C), d_model//4, L]
                feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)  # [(B*C), d_model//4]
                conv_features.append(feat)
            
            # 特征融合
            combined = torch.cat(conv_features, dim=-1)  # [(B*C), d_model//4*3]
            features = self.feature_fusion(combined)  # [(B*C), d_model]
            features = features.unsqueeze(-1)  # [(B*C), d_model, 1]
            
        else:
            # 标准模式
            x = self.revin(x, 'norm')
            B, L, C = x.shape
            x = x.permute(0, 2, 1)  # [B, C, L]
            
            features = []
            for i in range(C):
                channel_data = x[:, i:i+1, :]  # [B, 1, L]
                
                # 多尺度卷积特征提取
                conv_features = []
                for conv in self.conv1d_layers:
                    feat = F.gelu(conv(channel_data))  # [B, d_model//4, L]
                    feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)  # [B, d_model//4]
                    conv_features.append(feat)
                
                # 特征融合
                combined = torch.cat(conv_features, dim=-1)  # [B, d_model//4*3]
                feat = self.feature_fusion(combined)  # [B, d_model]
                features.append(feat)
                
            features = torch.stack(features, dim=-1)  # [B, d_model, C]
            
            # SE注意力增强
            features = self.se_block(features)
            
            # 自适应池化特征
            pooled_features = []
            for pool in self.adaptive_pools:
                pooled = pool(features)  # [B, d_model, pool_size]
                pooled = F.adaptive_avg_pool1d(pooled, 1).squeeze(-1)  # [B, d_model]
                pooled_features.append(pooled)
            
            # 池化特征融合
            pooled_combined = torch.cat(pooled_features, dim=-1)  # [B, d_model*4]
            pooled_enhanced = self.pool_fusion(pooled_combined)  # [B, d_model]
            
            # 频域特征
            try:
                freq_features = torch.fft.rfft(features.mean(dim=1), dim=-1)  # [B, seq_len//2+1]
                freq_magnitude = torch.abs(freq_features)
                if freq_magnitude.size(-1) != self.freq_enhance[0].in_features:
                    freq_magnitude = F.interpolate(
                        freq_magnitude.unsqueeze(1), 
                        size=self.freq_enhance[0].in_features, 
                        mode='linear', 
                        align_corners=False
                    ).squeeze(1)
                freq_enhanced = self.freq_enhance(freq_magnitude)  # [B, d_model]
                
                # 特征融合
                features = features + pooled_enhanced.unsqueeze(-1) + freq_enhanced.unsqueeze(-1)
            except:
                # 如果频域增强失败，只使用池化特征
                features = features + pooled_enhanced.unsqueeze(-1)
        
        # 计算重要性权重
        if features.dim() == 3:
            importance = self.importance_layer(features.permute(0, 2, 1))  # [B, C, 1]
            importance = importance.squeeze(-1)  # [B, C]
        else:
            importance = self.importance_layer(features.squeeze(-1))  # [(B*C), 1]
            importance = importance.squeeze(-1)  # [(B*C),]
        
        return features, importance

class AdaptiveMovingAvg(nn.Module):
    """自适应移动平均 - 可学习的核大小"""
    def __init__(self, kernel_sizes=[12, 24, 36], stride=1):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.stride = stride
        
        # 可学习的权重
        self.weights = nn.Parameter(torch.ones(len(kernel_sizes)))
        
        # 移动平均层
        self.moving_avgs = nn.ModuleList([
            nn.AvgPool1d(kernel_size=k, stride=stride, padding=0)
            for k in kernel_sizes
        ])
        
    def forward(self, x):
        # x: [B, C, L]
        trends = []
        
        for i, (kernel_size, avg_pool) in enumerate(zip(self.kernel_sizes, self.moving_avgs)):
            # 填充
            front = x[:, :, 0:1].repeat(1, 1, kernel_size - 1 - math.floor((kernel_size - 1) // 2))
            end = x[:, :, -1:].repeat(1, 1, math.floor((kernel_size - 1) // 2))
            padded_x = torch.cat([front, x, end], dim=-1)
            
            # 移动平均
            trend = avg_pool(padded_x)
            trends.append(trend)
        
        # 加权融合
        weights = F.softmax(self.weights, dim=0)
        combined_trend = sum(w * t for w, t in zip(weights, trends))
        
        return combined_trend

class EnhancedSeriesDecomp(nn.Module):
    """增强的序列分解"""
    def __init__(self, kernel_sizes=[12, 24, 36]):
        super().__init__()
        self.moving_avg = AdaptiveMovingAvg(kernel_sizes)
        
    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean

class EnhancedMultiHeadChannelAttention(nn.Module):
    """增强多头通道注意力机制"""
    def __init__(self, d_model, n_heads=8, dropout=0.1, max_channels=100):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0
        
        # 投影层
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, max_channels, d_model) * 0.02)
        
        # 温度缩放
        self.temperature = nn.Parameter(torch.ones(1))
        
        # 层归一化
        self.pre_norm = nn.LayerNorm(d_model)
        self.post_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # 相对位置偏置
        self.relative_bias = nn.Parameter(torch.zeros(2 * max_channels - 1))
        
    def forward(self, x):
        # x: [B, C, d_model]
        B, C, d_model = x.shape
        
        # 残差连接
        residual = x
        
        # 预归一化
        x = self.pre_norm(x)
        
        # 添加位置编码
        if C <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :C, :]
        else:
            # 如果通道数超过预设，使用插值
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2), 
                size=C, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            x = x + pos_enc
        
        # 多头注意力
        Q = self.q_linear(x).view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算 + 温度缩放
        scale = (self.head_dim ** -0.5) * torch.clamp(self.temperature, min=0.1, max=10.0)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # 相对位置偏置
        if C <= len(self.relative_bias) // 2 + 1:
            bias_indices = torch.arange(C, device=x.device).unsqueeze(0) - torch.arange(C, device=x.device).unsqueeze(1)
            bias_indices = bias_indices + len(self.relative_bias) // 2
            relative_bias = self.relative_bias[bias_indices]
            scores = scores + relative_bias.unsqueeze(0).unsqueeze(0)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, C, d_model)
        
        # 输出投影
        output = self.out_linear(attn_output)
        output = self.proj_dropout(output)
        
        # 门控机制
        gate_weights = self.gate(output)
        output = output * gate_weights
        
        # 残差连接和后归一化
        output = self.post_norm(residual + output)
        
        return output

class OptimizedChannelTransformer(nn.Module):
    """优化的通道Transformer"""
    def __init__(self, config):
        super().__init__()
        self.n_vars = config.enc_in
        self.d_model = config.d_model
        
        # 增强多头通道注意力
        self.channel_attention = EnhancedMultiHeadChannelAttention(
            config.d_model, 
            n_heads=min(8, config.d_model // 64),
            dropout=config.dropout,
            max_channels=config.enc_in
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.pred_len)
        )
        
    def forward(self, temporal_feature, input_data):
        # temporal_feature: [B, C, d_model]
        
        if self.n_vars > 1:
            # 通道注意力
            attended_features = self.channel_attention(temporal_feature)
            
            # 前馈网络
            ffn_output = self.ffn(attended_features)
            output_features = self.layer_norm(attended_features + ffn_output)
            
            # 预测
            output = self.prediction_head(output_features)
        else:
            # 单变量情况
            output = self.prediction_head(temporal_feature)
            
        return output

class EnhancedAdaptiveFusion(nn.Module):
    """增强自适应融合模块 - 动态权重和门控机制"""
    def __init__(self, pred_len, enc_in, num_sources=2, d_model=512):
        super().__init__()
        self.num_sources = num_sources
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        
        # 动态权重网络
        self.weight_network = nn.Sequential(
            nn.Linear(d_model * num_sources, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_sources),
            nn.Softmax(dim=-1)
        )
        
        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * num_sources, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # 特征增强网络
        self.feature_enhancer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 残差缩放
        self.residual_scale = nn.Parameter(torch.ones(num_sources))
        
        # 温度参数
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, *features):
        # features: 多个 [B, L, C] 张量
        assert len(features) == self.num_sources
        
        # 确保所有特征具有相同的形状
        target_shape = features[0].shape
        aligned_features = []
        
        for feat in features:
            if feat.shape != target_shape:
                # 如果形状不匹配，进行插值调整
                B, L, C = feat.shape
                target_B, target_L, target_C = target_shape
                
                if L != target_L:
                    # 时间维度调整
                    feat = feat.permute(0, 2, 1)  # [B, C, L]
                    feat = F.interpolate(feat, size=target_L, mode='linear', align_corners=False)
                    feat = feat.permute(0, 2, 1)  # [B, L, C]
                
                if C != target_C:
                    # 特征维度调整 - 截断或填充
                    if C > target_C:
                        feat = feat[:, :, :target_C]
                    else:
                        padding = torch.zeros(B, target_L, target_C - C, device=feat.device, dtype=feat.dtype)
                        feat = torch.cat([feat, padding], dim=-1)
            
            aligned_features.append(feat)
        
        B, L, C = aligned_features[0].shape
        
        # 特征增强
        enhanced_features = []
        for feat in aligned_features:
            # 全局平均池化获得特征表示
            global_feat = feat.mean(dim=1)  # [B, C]
            
            # 如果特征维度不匹配，进行调整
            if C != self.d_model:
                if C > self.d_model:
                    global_feat = global_feat[:, :self.d_model]
                else:
                    padding = torch.zeros(B, self.d_model - C, device=feat.device, dtype=feat.dtype)
                    global_feat = torch.cat([global_feat, padding], dim=-1)
            
            enhanced_feat = self.feature_enhancer(global_feat)  # [B, d_model]
            enhanced_features.append(enhanced_feat)
        
        # 动态权重计算
        combined_features = torch.cat(enhanced_features, dim=-1)  # [B, d_model * num_sources]
        dynamic_weights = self.weight_network(combined_features)  # [B, num_sources]
        
        # 温度缩放
        temperature = torch.clamp(self.temperature, min=0.1, max=10.0)
        dynamic_weights = F.softmax(dynamic_weights / temperature, dim=-1)
        
        # 门控机制
        gate_weights = self.gate_network(combined_features)  # [B, 1]
        
        # 加权融合
        residual_weights = F.softmax(self.residual_scale, dim=0)
        
        fused_output = torch.zeros_like(aligned_features[0])
        for i, feat in enumerate(aligned_features):
            weight = dynamic_weights[:, i:i+1].unsqueeze(-1)  # [B, 1, 1]
            residual_weight = residual_weights[i]
            fused_output = fused_output + weight * feat * residual_weight
        
        # 应用门控
        fused_output = fused_output * gate_weights.unsqueeze(-1)
        
        # 残差连接（与第一个特征）
        fused_output = fused_output + 0.1 * aligned_features[0]
        
        return fused_output

class Model(nn.Module):
    """
    YangDU2 优化版: 融合DUET和AdaWaveNet的高性能混合模型
    
    主要优化:
    1. 增强的特征提取器 - 多尺度卷积 + SE注意力
    2. 自适应序列分解 - 可学习的移动平均核
    3. 优化的通道Transformer - 多头注意力机制
    4. 自适应融合 - 学习最优融合权重
    5. 正则化和稳定性优化
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # 配置参数
        self.CI = getattr(configs, 'CI', True)
        self.dropout = configs.dropout
        
        # 核心组件初始化
        self._init_core_components(configs)
        self._init_wavelet_components(configs)
        self._init_transformer_components(configs)
        self._init_projection_layers(configs)
        
        # 增强自适应融合模块
        self.adaptive_fusion = EnhancedAdaptiveFusion(
            configs.pred_len, 
            configs.enc_in, 
            num_sources=2, 
            d_model=configs.d_model
        )
        
        # 知识蒸馏（可选）
        self.use_distillation = getattr(configs, 'use_distillation', False)
        if self.use_distillation:
            self.teacher_network = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model // 2),
                nn.GELU(),
                nn.Linear(configs.d_model // 2, configs.pred_len)
            )
            self.distillation_weight = nn.Parameter(torch.tensor(0.3))
        
        # 正则化
        self.apply(self._init_weights)
        
        self.register_buffer('clusters', None)

    def _init_core_components(self, configs):
        """初始化核心组件"""
        # AdaWaveNet组件
        self.kmeans = KMeans(n_clusters=configs.n_clusters)
        self.series_decomp = EnhancedSeriesDecomp()
        self.rev_seasonal = RevIN(configs.enc_in)
        self.rev_trend = RevIN(configs.enc_in)
        
        # 聚类线性层
        self.trend_linear = nn.ModuleDict({
            str(cluster_id): nn.Sequential(
                nn.Linear(self.seq_len, self.seq_len * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.seq_len * 2, configs.pred_len)
            ) for cluster_id in range(configs.n_clusters)
        })
        
        # DUET组件
        self.cluster_extractor = EnhancedLinearExtractor(configs)
        self.channel_transformer = OptimizedChannelTransformer(configs)
        
    def _init_wavelet_components(self, configs):
        """初始化小波组件"""
        self.encoder_levels = nn.ModuleList()
        self.linear_levels = nn.ModuleList()
        self.coef_linear_levels = nn.ModuleList()
        self.coef_dec_levels = nn.ModuleList()
        
        input_size = self.seq_len
        for i in range(configs.lifting_levels):
            # 小波编码器
            self.encoder_levels.add_module(
                f'encoder_level_{i}',
                AdpWaveletBlock(configs, input_size)
            )
            input_size = input_size // 2
            
            # 线性层
            enhanced_linear = nn.Sequential(
                nn.Linear(input_size, input_size * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(input_size * 2, input_size)
            )
            
            self.linear_levels.add_module(f'linear_level_{i}', enhanced_linear)
            self.coef_linear_levels.add_module(f'coef_linear_level_{i}', enhanced_linear)
            self.coef_dec_levels.add_module(f'coef_dec_level_{i}', enhanced_linear)

        self.input_size = input_size
        
        # 解码器
        self.decoder_levels = nn.ModuleList()
        for i in range(configs.lifting_levels-1, -1, -1):
            self.decoder_levels.add_module(
                f'decoder_level_{i}',
                InverseAdpWaveletBlock(configs, input_size=input_size)
            )
            input_size *= 2
        
    def _init_transformer_components(self, configs):
        """初始化Transformer组件"""
        # 嵌入层
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len // (2 ** configs.lifting_levels), 
            configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        
        # Transformer编码器
        self.encoder = Encoder(
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
        
    def _init_projection_layers(self, configs):
        """初始化投影层"""
        self.lowrank_projection = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.GELU(),
            nn.Linear(configs.d_model // 2, self.seq_len // (2 ** configs.lifting_levels))
        )
        
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Sequential(
                nn.Linear(self.seq_len, self.seq_len * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.seq_len * 2, configs.pred_len)
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(self.seq_len, self.seq_len * 2),
                nn.GELU(),
                nn.Linear(self.seq_len * 2, self.seq_len)
            )
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, clusters):
        # 增强的序列分解
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        
        # 归一化 - 避免就地操作
        x_enc = x.permute(0,2,1)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev  # 避免就地操作
        _, _, N = x_enc.shape

        moving_mean = self.rev_trend(moving_mean, 'norm')
        
        # DUET路径 - 增强的特征提取
        if self.CI:
            channel_independent_input = rearrange(x_enc, 'b l c -> (b c) l 1')
            temporal_feature, L_importance = self.cluster_extractor(channel_independent_input)
            temporal_feature = rearrange(temporal_feature, '(b c) d 1 -> b c d', b=x_enc.shape[0])
        else:
            temporal_feature, L_importance = self.cluster_extractor(x_enc)
            temporal_feature = rearrange(temporal_feature, 'b d c -> b c d')
        
        # 优化的通道Transformer
        duet_output = self.channel_transformer(temporal_feature, x_enc)
        
        # AdaWaveNet路径 - 小波编码
        x_enc_wavelet = x_enc.permute(0,2,1)
        encoded_coefficients = []
        x_embedding_levels = []
        coef_embedding_levels = []
        
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc_wavelet, r, details = l(x_enc_wavelet)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc_wavelet))
        
        # Transformer编码
        x_enc_wavelet = x_enc_wavelet.permute(0,2,1)
        enc_out = self.enc_embedding(x_enc_wavelet, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        
        # 小波解码
        for dec, x_emb_level, coef_emb_level, c_linear in zip(
            self.decoder_levels, 
            x_embedding_levels[::-1], 
            coef_embedding_levels[::-1], 
            self.coef_dec_levels[::-1]
        ):
            details = encoded_coefficients.pop()
            details = coef_emb_level + c_linear(details)
            x_dec = x_dec + x_emb_level
            x_dec = dec(x_dec, details)
        
        # AdaWaveNet输出
        wavelet_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        
        # 格式对齐
        duet_output = rearrange(duet_output, 'b c d -> b d c')
        
        # 自适应融合
        fused_output = self.adaptive_fusion(wavelet_out, duet_output)
        
        # 反归一化 - 避免就地操作
        stdev_expanded = stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        means_expanded = means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        fused_output = fused_output * stdev_expanded
        fused_output = fused_output + means_expanded
        
        # 趋势预测 - 增强的聚类线性层
        moving_mean_out = []
        for channel in range(moving_mean.shape[-1]):
            cluster_id = str(clusters[channel].item())
            channel_data = moving_mean[:, :, channel:channel+1].permute(0, 2, 1)
            trend_pred = self.trend_linear[cluster_id](channel_data).permute(0, 2, 1)
            moving_mean_out.append(trend_pred)
        moving_mean_out = torch.cat(moving_mean_out, dim=-1)
        moving_mean_out = self.rev_trend(moving_mean_out, 'denorm')
        
        # 最终输出
        final_output = fused_output + moving_mean_out
        
        # 知识蒸馏（如果启用）
        if self.use_distillation and self.training:
            # 使用全局平均池化的特征作为教师信号
            teacher_features = fused_output.mean(dim=1)  # [B, C]
            batch_size = teacher_features.size(0)
            
            # 调整特征维度以匹配教师网络
            if teacher_features.size(-1) != self.d_model:
                if teacher_features.size(-1) > self.d_model:
                    teacher_features = teacher_features[:, :self.d_model]
                else:
                    padding = torch.zeros(batch_size, self.d_model - teacher_features.size(-1), 
                                        device=teacher_features.device, dtype=teacher_features.dtype)
                    teacher_features = torch.cat([teacher_features, padding], dim=-1)
            
            teacher_output = self.teacher_network(teacher_features).unsqueeze(-1).repeat(1, 1, self.enc_in)
            teacher_output = teacher_output.view(batch_size, self.pred_len, self.enc_in)
            
            # 加权融合学生和教师输出
            distill_weight = torch.sigmoid(self.distillation_weight)
            final_output = distill_weight * final_output + (1 - distill_weight) * teacher_output
        
        return final_output, L_importance

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, C = x_enc.shape
        x_cluster = x_enc.permute(2,0,1).view(C, B * L)
        
        if self.clusters is None:
            clusters = self.kmeans.fit_predict(x_cluster)
            self.clusters = clusters
        else:
            clusters = self.clusters
            
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out, L_importance = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, clusters)
            return dec_out[:, -self.pred_len:, :]
        
        return None

# 辅助类定义
class AdpWaveletBlock(nn.Module):
    """AdaWaveNet的小波块"""
    def __init__(self, configs, input_size):
        super(AdpWaveletBlock, self).__init__()
        self.regu_details = getattr(configs, 'regu_details', 0.0)
        self.regu_approx = getattr(configs, 'regu_approx', 0.0)

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
    """AdaWaveNet的逆小波块"""
    def __init__(self, configs, input_size):
        super(InverseAdpWaveletBlock, self).__init__()
        self.inverse_wavelet = InverseLiftingScheme(configs.enc_in, input_size=input_size, kernel_size=configs.lifting_kernel_size)

    def forward(self, c, d):
        reconstructed = self.inverse_wavelet(c, d)
        return reconstructed