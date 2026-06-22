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
from collections import OrderedDict

def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)

class SKAttention1D(nn.Module):
    """
    SK注意力机制 - 适配1D时序数据
    基于Selective Kernel Networks，专门为时间序列数据优化
    """
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=8, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        
        # 多尺度1D卷积层 - 适配时序数据
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv1d(channel, channel, kernel_size=k, padding=k//2, groups=group)),
                    ('bn', nn.BatchNorm1d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        
        # 全局特征压缩
        self.fc = nn.Linear(channel, self.d)
        
        # 每个尺度对应的全连接层
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        """
        SK注意力前向传播 - 适配1D时序数据
        Args:
            x: 输入张量 (B, C, L) - 批次大小, 通道数, 序列长度
        Returns:
            SK增强的输出 (B, C, L)
        """
        # (B, C, L)
        B, C, L = x.size()
        
        # Split: 执行K个尺度对应的卷积操作
        conv_outs = []
        for conv in self.convs:
            scale = conv(x)  # 每个尺度的输出: (B, C, L)
            conv_outs.append(scale)
        
        feats = torch.stack(conv_outs, 0)  # (K, B, C, L)

        # Fuse: 多尺度信息融合
        U = sum(conv_outs)  # (B, C, L)
        
        # 全局平均池化: (B, C, L) -> (B, C)
        S = U.mean(-1)  # 在时序维度上进行平均池化
        
        # 降维提高计算效率: (B, C) -> (B, d)
        Z = self.fc(S)

        # 计算每个尺度的注意力权重
        weights = []
        for fc in self.fcs:
            weight = fc(Z)  # (B, d) -> (B, C)
            weights.append(weight.view(B, C, 1))  # (B, C, 1)
        
        scale_weight = torch.stack(weights, 0)  # (K, B, C, 1)
        scale_weight = self.softmax(scale_weight)  # 在尺度维度上执行softmax

        # Select: 加权融合多尺度特征
        V = (scale_weight * feats).sum(0)  # (B, C, L)
        return V

class MultiScaleSKConv1D(nn.Module):
    """
    多尺度SK卷积模块 - 专门为时序数据设计
    """
    def __init__(self, channels, kernels=[3, 5, 7, 9], reduction=8, dropout=0.1):
        super().__init__()
        
        # SK注意力机制
        self.sk_attention = SKAttention1D(
            channel=channels, 
            kernels=kernels, 
            reduction=max(4, channels // reduction)
        )
        
        # 特征增强
        self.feature_enhance = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.InstanceNorm1d(channels)
        
        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        """
        Args:
            x: 输入张量 (B, C, L)
        Returns:
            SK增强的多尺度融合特征 (B, C, L)
        """
        # SK注意力多尺度特征选择
        sk_enhanced = self.sk_attention(x)
        
        # 特征增强
        enhanced = self.feature_enhance(sk_enhanced)
        
        # 自适应残差连接
        output = self.residual_weight * enhanced + (1 - self.residual_weight) * x
        output = self.dropout(output)
        output = self.norm(output)
        
        return output

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
        device = x.device
        window_sum = torch.nn.functional.conv1d(x_padded, 
                                                weight=torch.ones((1, num_channels, self.kernel_size), device=device),
                                                stride=self.stride)

        # Count non-zero elements in each window
        window_count = torch.nn.functional.conv1d(non_zero_mask.float(), 
                                                  weight=torch.ones((1, num_channels, self.kernel_size), device=device),
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

class SKEnhancedAdpWaveletBlock(nn.Module):
    """
    SK增强的自适应小波块
    在小波变换中集成SK注意力机制进行多尺度特征选择
    """
    def __init__(self, configs, input_size):
        super(SKEnhancedAdpWaveletBlock, self).__init__()
        self.regu_details = configs.regu_details
        self.regu_approx = configs.regu_approx
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingScheme(configs.enc_in, k_size=configs.lifting_kernel_size, input_size=input_size)
        self.norm_x = normalization(configs.enc_in)
        self.norm_d = normalization(configs.enc_in)
        
        # SK注意力机制 - 输入特征多尺度选择
        self.input_sk = MultiScaleSKConv1D(
            channels=configs.enc_in, 
            kernels=[3, 5, 7, 9],  # 多个时序尺度
            reduction=max(4, configs.enc_in // 8),
            dropout=configs.dropout
        )
        
        # SK注意力机制 - 小波系数多尺度选择
        self.coef_sk = MultiScaleSKConv1D(
            channels=configs.enc_in,
            kernels=[3, 5, 7],  # 适合小波系数的尺度
            reduction=max(4, configs.enc_in // 8),
            dropout=configs.dropout
        )
        
        self.detail_sk = MultiScaleSKConv1D(
            channels=configs.enc_in,
            kernels=[3, 5, 7],
            reduction=max(4, configs.enc_in // 8),
            dropout=configs.dropout
        )

    def forward(self, x):
        # 输入特征SK多尺度选择增强
        x_enhanced = self.input_sk(x)
        
        # 小波变换
        (c, d) = self.wavelet(x_enhanced)
        
        # 小波系数SK多尺度选择增强
        c = self.coef_sk(c)
        d = self.detail_sk(d)
        
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

class SKEnhancedClusteredLinear(nn.Module):
    """
    SK增强的聚类线性层
    在聚类线性变换中集成SK注意力机制
    """
    def __init__(self, n_clusters, enc_in, seq_len, pred_len):
        super().__init__()
        self.n_clusters = n_clusters
        self.enc_in = enc_in
        
        # 增强的线性层 - 添加非线性激活
        self.linear_layers = nn.ModuleDict({
            str(cluster_id): nn.Sequential(
                nn.Linear(seq_len, seq_len * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(seq_len * 2, pred_len)
            ) for cluster_id in range(n_clusters)
        })
        
        # SK注意力机制 - 输入特征多尺度选择
        self.input_sk = MultiScaleSKConv1D(
            channels=enc_in,
            kernels=[3, 5, 7, 9],
            reduction=max(4, enc_in // 8),
            dropout=0.1
        )
        
        # SK注意力机制 - 输出特征多尺度选择
        self.output_sk = MultiScaleSKConv1D(
            channels=enc_in,
            kernels=[3, 5, 7],
            reduction=max(4, enc_in // 8),
            dropout=0.1
        )
        
    def forward(self, x, clusters):
        # 输入SK多尺度选择增强
        x_enhanced = self.input_sk(x)
        
        output = []
        assert self.enc_in == len(clusters)
        for channel in range(self.enc_in):
            cluster_id = str(clusters[channel].item())
            channel_data = x_enhanced[:, channel, :].unsqueeze(1)
            transformed_channel = self.linear_layers[cluster_id](channel_data)
            output.append(transformed_channel)
        output = torch.concat(output, dim=1)
        
        # 输出SK多尺度选择增强
        output = self.output_sk(output)
        
        return output

class Model(nn.Module):
    """
    SK增强的AdaWaveNet模型 (YangC)
    将SK注意力机制深度集成到AdaWaveNet的各个关键组件中
    基于Selective Kernel Networks进行多尺度特征选择
    Paper link: https://arxiv.org/abs/2310.06625
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
        
        # 使用SK增强的聚类线性层
        self.trend_linear = SKEnhancedClusteredLinear(configs.n_clusters, configs.enc_in, self.seq_len, configs.pred_len)
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len // (2 ** configs.lifting_levels), configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # 使用SK增强的小波块构建编码器层级
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
            # 使用SK增强的小波块
            self.encoder_levels.add_module(
                'encoder_level_'+str(i),
                SKEnhancedAdpWaveletBlock(configs, input_size)
            )
            in_planes *= 1
            input_size = input_size // 2 
            
            # 增强的线性变换层
            enhanced_linear = nn.Sequential(
                nn.Linear(input_size, input_size * 2),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(input_size * 2, input_size * expand_ratio)
            )
            
            self.linear_levels.add_module('linear_level_'+str(i), enhanced_linear)
            self.coef_linear_levels.add_module('linear_level_'+str(i), enhanced_linear)
            self.coef_dec_levels.add_module('linear_level_'+str(i), enhanced_linear)

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
        
        # 增强的低秩投影
        if self.task_name == "super_resolution":
            self.lowrank_projection = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model * 2),
                nn.GELU(),
                nn.Linear(configs.d_model * 2, self.pred_len // (2 ** configs.lifting_levels))
            )
        else:
            self.lowrank_projection = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model * 2),
                nn.GELU(),
                nn.Linear(configs.d_model * 2, self.seq_len // (2 ** configs.lifting_levels))
            )

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
        
        # 增强的投影层
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Sequential(
                nn.Linear(self.seq_len, self.seq_len * 2),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.seq_len * 2, configs.pred_len)
            )
        elif self.task_name == 'imputation':
            self.projection = nn.Sequential(
                nn.Linear(self.seq_len, self.seq_len * 2),
                nn.GELU(),
                nn.Linear(self.seq_len * 2, self.seq_len)
            )
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Sequential(
                nn.Linear(self.seq_len, self.seq_len * 2),
                nn.GELU(),
                nn.Linear(self.seq_len * 2, self.seq_len)
            )
        elif self.task_name == 'super_resolution':
            self.projection = nn.Sequential(
                nn.Linear(configs.pred_len, configs.pred_len * 2),
                nn.GELU(),
                nn.Linear(configs.pred_len * 2, configs.pred_len)
            )

        self.register_buffer('clusters', None)
        
        # 全局SK注意力 - 应用于最终输出
        self.global_sk = MultiScaleSKConv1D(
            channels=configs.enc_in,
            kernels=[3, 5, 7, 9],
            reduction=max(4, configs.enc_in // 8),
            dropout=configs.dropout
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, clusters):
        # 序列分解
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        
        # 归一化
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
        
        # 编码过程 - 使用SK增强的小波块
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
            
        # 嵌入
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
            
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        
        # 反归一化
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        # 趋势预测 - 使用SK增强的聚类线性层
        moving_mean_out = self.trend_linear(moving_mean.permute(0,2,1), self.clusters).permute(0,2,1)
        moving_mean_out = self.rev_trend(moving_mean_out, 'denorm')
        
        # 最终输出 - 应用全局SK注意力
        dec_out = dec_out + moving_mean_out
        dec_out = dec_out.permute(0,2,1)  # (B, C, L)
        dec_out = self.global_sk(dec_out)  # 全局SK多尺度选择增强
        dec_out = dec_out.permute(0,2,1)  # (B, L, C)
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # 复用forecast逻辑
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




