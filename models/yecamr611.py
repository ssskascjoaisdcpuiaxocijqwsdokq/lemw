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
from torch.nn import init

def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)

class AdaptiveMultiDWConv1D(nn.Module):
    """
    优化版多感受野深度可分离卷积
    - 自适应多尺度分割
    - 动态权重调整
    - 增强的特征融合
    """
    def __init__(self, dim=768, num_scales=4):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        
        # 动态分割比例学习
        self.scale_ratios = nn.Parameter(torch.tensor([1.0, 0.75, 0.5, 0.25], dtype=torch.float32))
        
        # 多个深度可分离卷积分支，使用不同kernel size
        self.dwconv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3+2*i, padding=1+i, groups=dim, bias=False),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Conv1d(dim, dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(dim)
            ) for i in range(num_scales)
        ])
        
        # 自适应特征融合网络
        fusion_dim = max(1, dim // 4)
        self.fusion_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim * num_scales, fusion_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(fusion_dim, num_scales, 1),
            nn.Sigmoid()
        )
        
        # 特征增强模块
        self.feature_enhance = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim)
        )
        
        self.dropout = nn.Dropout(0.1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x, seq_len):
        """
        x: (B, L, C) - 时序特征
        seq_len: 序列长度
        """
        B, L, C = x.shape
        x = x.transpose(1, 2)  # (B, C, L)
        
        # 自适应多尺度特征提取
        scale_features = []
        normalized_ratios = F.softmax(self.scale_ratios, dim=0)
        
        for i, (branch, ratio) in enumerate(zip(self.dwconv_branches, normalized_ratios)):
            # 动态确定尺度长度
            scale_len = max(int(L * ratio), L // 8)  # 确保最小长度
            
            if scale_len < L:
                start_idx = (L - scale_len) // 2
                x_scale = x[:, :, start_idx:start_idx + scale_len]
            else:
                x_scale = x
            
            # 应用深度可分离卷积
            feat = branch(x_scale)
            
            # 插值到统一长度
            if feat.size(-1) != L:
                feat = F.interpolate(feat, size=L, mode='linear', align_corners=False)
            
            scale_features.append(feat)
        
        # 自适应权重融合
        concat_features = torch.cat(scale_features, dim=1)  # (B, C*num_scales, L)
        fusion_weights = self.fusion_net(concat_features)  # (B, num_scales, 1)
        fusion_weights = fusion_weights.unsqueeze(-1)  # (B, num_scales, 1, 1)
        
        # 加权融合多尺度特征
        fused_features = sum(w * feat for w, feat in zip(fusion_weights.unbind(1), scale_features))
        
        # 特征增强
        enhanced_features = self.feature_enhance(fused_features)
        
        # 残差连接
        output = x + enhanced_features + fused_features * 0.3
        output = self.dropout(output)
        
        return output.transpose(1, 2)  # (B, L, C)

class SpatioTemporalAttention1D(nn.Module):
    """
    时空联合注意力机制
    - 结合通道注意力和时序注意力
    - 自适应权重分配
    """
    def __init__(self, channel=512, seq_len=96, reduction=16, dropout=0.1):
        super().__init__()
        self.channel = channel
        self.seq_len = seq_len
        
        # 确保reduction后的通道数至少为1
        reduced_channel = max(1, channel // reduction)
        
        # 通道注意力分支
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, reduced_channel, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channel, channel, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 时序注意力分支
        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, seq_len)),
            nn.Conv2d(1, 1, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.Sigmoid()
        )
        
        # 联合注意力融合
        self.joint_fusion = nn.Sequential(
            nn.Conv1d(channel, channel // 2, 1, bias=False),
            nn.BatchNorm1d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channel // 2, channel, 1, bias=False),
            nn.BatchNorm1d(channel)
        )
        
        # 自适应权重学习
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: (B, C, L)
        """
        B, C, L = x.size()
        residual = x
        
        # 通道注意力
        channel_att = self.channel_attention(x)  # (B, C, 1)
        channel_enhanced = x * channel_att.expand_as(x)
        
        # 时序注意力
        x_temp = x.unsqueeze(1)  # (B, 1, C, L)
        temporal_att = self.temporal_attention(x_temp)  # (B, 1, 1, L)
        temporal_enhanced = x * temporal_att.squeeze(1).expand_as(x)
        
        # 联合特征融合
        joint_features = self.joint_fusion(channel_enhanced + temporal_enhanced)
        
        # 自适应权重分配
        weights = self.weight_net(x)  # (B, 2, 1)
        w1, w2 = weights[:, 0:1, :], weights[:, 1:2, :]
        
        # 加权融合
        output = residual + w1 * joint_features + w2 * (channel_enhanced + temporal_enhanced) * 0.5
        output = self.dropout(output)
        
        return output

class EnhancedMRFP1D(nn.Module):
    """
    增强版多感受野处理器
    - 简化重复功能
    - 增强残差连接
    - 添加特征重标定
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 输入投影
        self.input_proj = nn.Linear(in_features, hidden_features)
        
        # 多感受野处理（简化版）
        self.multi_scale_conv = AdaptiveMultiDWConv1D(hidden_features, num_scales=3)
        
        # 特征重标定模块
        recalibration_dim = max(1, hidden_features // 4)
        self.feature_recalibration = nn.Sequential(
            nn.Linear(hidden_features, recalibration_dim),
            nn.ReLU(inplace=True),
            nn.Linear(recalibration_dim, hidden_features),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_features, out_features)
        
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        
        # 残差缩放参数
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, seq_len):
        """
        x: (B, L, C)
        """
        residual = x
        
        # 输入投影
        x = self.input_proj(x)
        x = self.act(x)
        x = self.drop(x)
        
        # 多感受野处理
        x = self.multi_scale_conv(x, seq_len)
        
        # 特征重标定
        B, L, C = x.shape
        recalibration_weights = self.feature_recalibration(x.mean(dim=1))  # (B, C)
        x = x * recalibration_weights.unsqueeze(1)  # (B, L, C)
        
        # 输出投影
        x = self.output_proj(x)
        x = self.drop(x)
        
        # 增强残差连接
        if x.shape == residual.shape:
            x = residual + x * self.residual_scale
        
        return x

class DynamicClusteredLinear(nn.Module):
    """
    动态聚类线性层
    - 动态聚类更新
    - 聚类质量评估
    - 自适应权重调整
    """
    def __init__(self, n_clusters, enc_in, seq_len, pred_len, update_freq=100):
        super().__init__()
        self.n_clusters = n_clusters
        self.enc_in = enc_in
        self.update_freq = update_freq
        self.update_counter = 0
        
        # 为每个聚类定义线性层
        self.linear_layers = nn.ModuleDict({
            str(cluster_id): nn.Linear(seq_len, pred_len) for cluster_id in range(n_clusters)
        })
        
        # 聚类质量评估网络 - 使用自适应池化处理不同长度的输入
        self.quality_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(seq_len),  # 自适应池化到固定长度
            nn.Flatten(),
            nn.Linear(seq_len, max(1, seq_len // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, seq_len // 4), 1),
            nn.Sigmoid()
        )
        
        # 自适应权重网络
        self.adaptive_weights = nn.Parameter(torch.ones(n_clusters) / n_clusters)
        
        # 聚类中心缓存
        self.register_buffer('cluster_centers', None)
        self.register_buffer('cluster_quality', torch.ones(n_clusters))

    def update_clusters(self, x, clusters):
        """动态更新聚类"""
        self.update_counter += 1
        if self.update_counter % self.update_freq == 0:
            # 使用detach()避免梯度计算问题
            with torch.no_grad():
                # 重新计算聚类中心
                B, C, L = x.shape
                x_flat = x.view(C, B * L).detach()
                
                # 计算新的聚类中心
                new_centers = []
                for i in range(self.n_clusters):
                    mask = clusters == i
                    if mask.sum() > 0:
                        center = x_flat[mask].mean(dim=0)
                        new_centers.append(center)
                    else:
                        # 如果某个聚类为空，保持原中心
                        if self.cluster_centers is not None:
                            new_centers.append(self.cluster_centers[i])
                        else:
                            new_centers.append(torch.randn(B * L, device=x.device))
                
                self.cluster_centers = torch.stack(new_centers)
                
                # 评估聚类质量
                for i in range(self.n_clusters):
                    mask = clusters == i
                    if mask.sum() > 0:
                        cluster_data = x_flat[mask]  # (num_channels_in_cluster, B*L)
                        # 重塑为(1, num_channels, B*L)用于自适应池化
                        cluster_data_reshaped = cluster_data.unsqueeze(0).mean(dim=1, keepdim=True)  # (1, 1, B*L)
                        # 临时启用梯度计算用于quality_net
                        cluster_data_reshaped.requires_grad_(True)
                        quality = self.quality_net(cluster_data_reshaped).squeeze().detach()
                        self.cluster_quality[i] = quality

    def forward(self, x, clusters):
        """
        x: (B, C, L)
        clusters: (C,) 聚类标签
        """
        # 动态更新聚类（训练时）- 暂时禁用以避免梯度问题
        # if self.training:
        #     self.update_clusters(x, clusters)
        
        output = []
        normalized_weights = F.softmax(self.adaptive_weights, dim=0)
        
        for channel in range(self.enc_in):
            cluster_id = str(clusters[channel].item())
            channel_data = x[:, channel, :].unsqueeze(1)
            
            # 应用对应的线性变换
            transformed_channel = self.linear_layers[cluster_id](channel_data)
            
            # 根据聚类质量调整权重
            quality_weight = self.cluster_quality[clusters[channel]]
            weighted_channel = transformed_channel * quality_weight * normalized_weights[clusters[channel]]
            
            output.append(weighted_channel)
        
        output = torch.cat(output, dim=1)
        return output

class OptimizedWaveletBlock(nn.Module):
    """
    优化的小波变换块
    - 改进的正则化策略
    - 自适应核大小
    - 增强的特征提取
    """
    def __init__(self, configs, input_size):
        super(OptimizedWaveletBlock, self).__init__()
        self.regu_details = configs.regu_details
        self.regu_approx = configs.regu_approx
        
        # 自适应核大小
        adaptive_kernel_size = min(max(3, input_size // 32), 7)
        
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingScheme(configs.enc_in, k_size=adaptive_kernel_size, input_size=input_size)
        
        # 改进的正则化
        self.norm_x = nn.Sequential(
            normalization(configs.enc_in),
            nn.Dropout(0.05)
        )
        self.norm_d = nn.Sequential(
            normalization(configs.enc_in),
            nn.Dropout(0.05)
        )
        
        # 特征增强
        self.feature_enhance = nn.Conv1d(configs.enc_in, configs.enc_in, 
                                       kernel_size=3, padding=1, groups=configs.enc_in)
        
        # 自适应权重
        self.detail_weight = nn.Parameter(torch.ones(1) * 0.1)
        self.approx_weight = nn.Parameter(torch.ones(1) * 0.9)

    def forward(self, x):
        (c, d) = self.wavelet(x)
        
        # 特征增强
        c_enhanced = self.feature_enhance(c)
        c = c + c_enhanced * self.approx_weight
        
        r = None
        if(self.regu_approx + self.regu_details != 0.0):
            if self.regu_details:
                rd = self.regu_details * d.abs().mean() * self.detail_weight
            if self.regu_approx:
                rc = self.regu_approx * torch.dist(c.mean(), x.mean(), p=2) * self.approx_weight
            if self.regu_approx == 0.0:
                r = rd
            elif self.regu_details == 0.0:
                r = rc
            else:
                r = rd + rc

        x = self.norm_x(c)
        d = self.norm_d(d)
        
        return x, r, d

class EnhancedECABlock1D(nn.Module):
    """
    增强版ECA注意力块
    - 时空联合注意力
    - 改进的多感受野处理
    - 动态特征融合
    """
    def __init__(self, channel=512, seq_len=96, kernel_size=3, dropout=0.1, mrfp_ratio=2.0):
        super().__init__()
        self.channel = channel
        
        # 预归一化
        self.pre_norm = nn.LayerNorm(channel)
        
        # 时空联合注意力
        self.spatiotemporal_attention = SpatioTemporalAttention1D(
            channel=channel, seq_len=seq_len, dropout=dropout
        )
        
        # 增强版MRFP
        self.enhanced_mrfp = EnhancedMRFP1D(
            in_features=channel,
            hidden_features=int(channel * mrfp_ratio),
            drop=dropout
        )
        
        # 动态特征融合网络
        fusion_dim = max(1, channel // 4)
        self.dynamic_fusion = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, fusion_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(fusion_dim, 3, 1),  # 3个分支的权重
            nn.Softmax(dim=1)
        )
        
        # 特征增强网络
        self.feature_enhance = nn.Sequential(
            nn.Conv1d(channel, channel * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(channel * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channel * 2, channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(channel)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: (B, C, L)
        """
        B, C, L = x.size()
        residual = x
        
        # 预归一化
        x_norm = self.pre_norm(x.transpose(1, 2)).transpose(1, 2)
        
        # 时空联合注意力
        attention_out = self.spatiotemporal_attention(x_norm)
        
        # 增强版MRFP处理
        mrfp_input = attention_out.transpose(1, 2)
        mrfp_out = self.enhanced_mrfp(mrfp_input, L).transpose(1, 2)
        
        # 特征增强
        enhanced_out = self.feature_enhance(mrfp_out)
        
        # 动态权重融合
        fusion_weights = self.dynamic_fusion(x)  # (B, 3, 1)
        w1, w2, w3 = fusion_weights[:, 0:1, :], fusion_weights[:, 1:2, :], fusion_weights[:, 2:3, :]
        
        # 多分支动态融合
        output = (residual * w1 + 
                 attention_out * w2 + 
                 enhanced_out * w3)
        
        output = self.dropout(output)
        return output

# 保持原有的moving_avg, series_decomp, InverseAdpWaveletBlock等模块不变
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

class moving_avg_imputation(nn.Module):
    """
    Moving average block modified to ignore zeros in the moving window.
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg_imputation, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Padding on the both ends of time series
        # x - B, C, L
        num_channels = x.shape[1]
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2))
        end = x[:, :, -1:].repeat(1, 1, math.floor((self.kernel_size - 1) // 2))
        x_padded = torch.cat([front, x, end], dim=-1)

        # Create a mask for non-zero elements
        non_zero_mask = x_padded != 0

        # Calculate sum of non-zero elements in each window
        weight = torch.ones((1, num_channels, self.kernel_size), device=x_padded.device)
        window_sum = torch.nn.functional.conv1d(x_padded, 
                                                weight=weight,
                                                stride=self.stride)

        # Count non-zero elements in each window
        window_count = torch.nn.functional.conv1d(non_zero_mask.float(), 
                                                  weight=weight,
                                                  stride=self.stride)

        # Avoid division by zero; set count to 1 where there are no non-zero elements
        window_count = torch.clamp(window_count, min=1)

        # Compute the moving average
        moving_avg = window_sum / window_count
        return moving_avg

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size=24, stride=1, imputation=False):
        super(series_decomp, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.moving_avg = moving_avg(kernel_size, stride=stride) if not imputation else moving_avg_imputation(self.kernel_size, self.stride)

    def forward(self, x):
        moving_mean = self.moving_avg(x) 
        res = x - moving_mean
        return res, moving_mean

class InverseAdpWaveletBlock(nn.Module):
    def __init__(self, configs, input_size):
        super(InverseAdpWaveletBlock, self).__init__()
        self.inverse_wavelet = InverseLiftingScheme(configs.enc_in, input_size=input_size, kernel_size=configs.lifting_kernel_size)

    def forward(self, c, d):
        reconstructed = self.inverse_wavelet(c, d)
        return reconstructed

class Model(nn.Module):
    """
    YECAMR611模型: 全面优化版本
    - 自适应多尺度处理
    - 时空联合注意力
    - 动态聚类机制
    - 增强的小波变换
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
        self.series_decomp = series_decomp(imputation = self.task_name=='imputation')
        self.rev_seasonal = RevIN(configs.enc_in)
        self.rev_trend = RevIN(configs.enc_in)
        
        # 动态聚类线性层
        self.trend_linear = DynamicClusteredLinear(
            configs.n_clusters, configs.enc_in, self.seq_len, configs.pred_len
        )
        
        # 增强版ECA注意力模块
        self.enhanced_eca_block = EnhancedECABlock1D(
            channel=configs.enc_in, 
            seq_len=self.seq_len,
            kernel_size=getattr(configs, 'eca_kernel_size', 3),
            dropout=configs.dropout,
            mrfp_ratio=getattr(configs, 'mrfp_ratio', 2.0)
        )
        
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
                OptimizedWaveletBlock(configs, input_size)  # 使用优化版小波块
            )
            in_planes *= 1
            input_size = input_size // 2 
            self.linear_levels.add_module(
                'linear_level_'+str(i),
                nn.Sequential(
                    nn.Linear(input_size, input_size * expand_ratio),
                    nn.Dropout(0.05)  # 添加dropout
                )
            )
            self.coef_linear_levels.add_module(
                'linear_level_'+str(i),
                nn.Sequential(
                    nn.Linear(input_size, input_size * expand_ratio),
                    nn.Dropout(0.05)
                )
            )
            self.coef_dec_levels.add_module(
                'linear_level_'+str(i),
                nn.Sequential(
                    nn.Linear(input_size, input_size * expand_ratio),
                    nn.Dropout(0.05)
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
        
        # 应用增强版ECA注意力机制
        x = self.enhanced_eca_block(x)
        
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
        # Encoding
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
        
        # 应用增强版ECA注意力机制
        x = self.enhanced_eca_block(x)
        
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
        # Encoding
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

    def anomaly_detection(self, x_enc):
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        
        # 应用增强版ECA注意力机制
        x = self.enhanced_eca_block(x)
        
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
        # Encoding
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

    def super_resolution(self, x_enc):
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        
        # 应用增强版ECA注意力机制
        x = self.enhanced_eca_block(x)
        
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
        # Encoding
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
            if self.output_attention:
                return dec_out[:, -self.pred_len:, :], None  # [B, L, D], attention
            else:
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            if self.output_attention:
                return dec_out, None  # [B, L, D], attention
            else:
                return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            if self.output_attention:
                return dec_out, None  # [B, L, D], attention
            else:
                return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == 'super_resolution':
            dec_out = self.super_resolution(x_enc)
            if self.output_attention:
                return dec_out, None
            else:
                return dec_out
        return None
