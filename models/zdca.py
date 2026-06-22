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

class TemporalResidual(nn.Module):
    """
    时序残差连接 - 适配CMUNeXt的Residual到1D时序
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class TemporalDepthwiseConv(nn.Module):
    """
    时序深度卷积 - 高效的局部特征提取
    """
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, 
            kernel_size=kernel_size, 
            groups=channels, 
            padding=kernel_size // 2
        )
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        # x: (B, C, L)
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class TemporalInvertedBottleneck(nn.Module):
    """
    时序反向瓶颈 - 适配CMUNeXt的反向瓶颈设计到1D时序
    
    核心思想：
    1. 扩展中间维度 - 增强特征表达能力
    2. 深度可分离卷积 - 减少计算开销
    3. 残差连接 - 稳定训练过程
    """
    def __init__(self, channels, expansion_factor=4, kernel_size=7):
        super().__init__()
        expanded_channels = channels * expansion_factor
        
        # 反向瓶颈设计：先扩展再压缩
        self.expand = nn.Sequential(
            nn.Conv1d(channels, expanded_channels, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(expanded_channels)
        )
        
        # 深度卷积处理扩展后的特征
        self.depthwise = TemporalDepthwiseConv(expanded_channels, kernel_size)
        
        # 压缩回原始维度
        self.compress = nn.Sequential(
            nn.Conv1d(expanded_channels, channels, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(channels)
        )
        
    def forward(self, x):
        # x: (B, C, L)
        # 扩展
        expanded = self.expand(x)
        
        # 深度卷积
        processed = self.depthwise(expanded)
        
        # 压缩
        compressed = self.compress(processed)
        
        return compressed

class TemporalCMUNeXtBlock(nn.Module):
    """
    时序CMUNeXt块 - 适配CMUNeXtBlock到1D时序数据
    
    核心特性：
    1. 高效全局信息提取 - 大卷积核捕获长距离依赖
    2. 轻量化设计 - 深度可分离卷积减少参数
    3. 反向瓶颈 - 增强特征表达能力
    4. 残差连接 - 稳定训练过程
    """
    def __init__(self, channels, depth=2, kernel_size=7, expansion_factor=4):
        super().__init__()
        
        # 多层时序处理块
        self.blocks = nn.ModuleList([
            nn.Sequential(
                # 深度卷积残差块
                TemporalResidual(
                    TemporalDepthwiseConv(channels, kernel_size)
                ),
                # 反向瓶颈块
                TemporalInvertedBottleneck(channels, expansion_factor, kernel_size)
            ) for _ in range(depth)
        ])
        
        # 输出归一化
        self.output_norm = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        # x: (B, C, L)
        for block in self.blocks:
            x = block(x)
        
        x = self.output_norm(x)
        return x

class AdaptiveTemporalFusion(nn.Module):
    """
    自适应时序融合 - 智能融合多种特征
    """
    def __init__(self, channels):
        super().__init__()
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        
        # 局部特征增强
        self.local_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # x: (B, C, L)
        identity = x
        
        # 全局特征
        global_feat = self.global_pool(x).squeeze(-1)  # (B, C)
        global_weight = self.global_fc(global_feat).unsqueeze(-1)  # (B, C, 1)
        global_enhanced = x * global_weight
        
        # 局部特征
        local_enhanced = self.local_conv(x)
        
        # 自适应融合
        weight = torch.sigmoid(self.fusion_weight)
        fused = weight * global_enhanced + (1 - weight) * local_enhanced
        
        # 残差连接
        output = fused + identity
        
        return output

class CMUNeXtEnhancedEncoder(nn.Module):
    """
    CMUNeXt增强编码器 - 集成时序CMUNeXt到AdaWaveNet编码器中
    """
    def __init__(self, configs):
        super(CMUNeXtEnhancedEncoder, self).__init__()
        
        # CMUNeXt特征提取块
        self.cmunext_block = TemporalCMUNeXtBlock(
            channels=configs.d_model,
            depth=2,
            kernel_size=7,
            expansion_factor=4
        )
        
        # 自适应特征融合
        self.adaptive_fusion = AdaptiveTemporalFusion(configs.d_model)
        
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
        
        # 特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.LayerNorm(configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout)
        )
        
        # 动态权重学习
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(configs.d_model, 2, 1),
            nn.Softmax(dim=1)
        )
        
        # 输出处理
        self.output_norm = nn.LayerNorm(configs.d_model)
        
    def forward(self, x, attn_mask=None):
        # x: (B, L, C)
        identity = x
        
        # 转换为卷积格式进行CMUNeXt处理
        x_conv = x.transpose(1, 2)  # (B, C, L)
        
        # CMUNeXt特征提取
        cmunext_out = self.cmunext_block(x_conv)  # (B, C, L)
        
        # 自适应特征融合
        fused_conv = self.adaptive_fusion(cmunext_out)  # (B, C, L)
        
        # 转换回序列格式
        cmunext_features = fused_conv.transpose(1, 2)  # (B, L, C)
        
        # Transformer处理
        transformer_out, attns = self.transformer_encoder(x, attn_mask=attn_mask)
        
        # 动态权重计算
        weights = self.weight_net(x_conv)  # (B, 2, 1)
        weights = weights.squeeze(-1).unsqueeze(1)  # (B, 1, 2)
        
        # 加权融合
        weighted_cmunext = cmunext_features * weights[:, :, 0:1]
        weighted_transformer = transformer_out * weights[:, :, 1:2]
        
        # 特征拼接和融合
        concat_features = torch.cat([weighted_cmunext, weighted_transformer], dim=-1)
        fused_output = self.feature_fusion(concat_features)
        
        # 残差连接和归一化
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
    ZDCA: AdaWaveNet + CMUNeXt时序高效特征提取
    
    主要创新:
    1. 时序CMUNeXt块 - 高效全局信息提取和轻量化设计
    2. 反向瓶颈机制 - 增强特征表达能力
    3. 深度可分离卷积 - 减少计算开销
    4. 自适应特征融合 - 智能融合多种特征
    5. 动态权重学习 - 根据输入自适应调整
    
    核心优势:
    - 大卷积核捕获长距离时序依赖
    - 反向瓶颈增强特征表达
    - 深度卷积减少参数量
    - 残差连接稳定训练
    - 自适应融合提升性能
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

        # CMUNeXt增强编码器
        self.encoder = CMUNeXtEnhancedEncoder(configs)
        
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
        
        # CMUNeXt增强编码器处理
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




