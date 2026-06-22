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

class AdaptiveSEAttention1D(nn.Module):
    """
    自适应SE注意力机制 - 针对96/192预测窗口优化
    """
    def __init__(self, channel=512, reduction=16, num_inputs=3, pred_len=96):
        super().__init__()
        self.num_inputs = num_inputs
        self.pred_len = pred_len
        
        # 双重池化：平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 根据预测长度自适应调整reduction
        if pred_len <= 96:
            self.reduction = max(4, min(reduction, channel // 4))
        else:  # pred_len == 192
            self.reduction = max(8, min(reduction, channel // 6))
        
        # 增强的SE网络
        self.fc = nn.Sequential(
            nn.Linear(channel * 2, channel // self.reduction, bias=False),  # *2因为concat了avg和max
            nn.LayerNorm(channel // self.reduction),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(channel // self.reduction, num_inputs * channel, bias=False),
        )
        
        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.2))
        
    def forward(self, *inputs):
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")
            
        B, C, L = inputs[0].size()
        
        # 融合所有输入
        x_combined = sum(inputs)
        
        # 双重池化
        avg_out = self.avg_pool(x_combined).view(B, C)
        max_out = self.max_pool(x_combined).view(B, C)
        
        # 特征融合
        y = torch.cat([avg_out, max_out], dim=1)  # (B, 2C)
        
        # SE变换
        y = self.fc(y).view(B, self.num_inputs * C, 1)
        
        # 分割权重并应用sigmoid激活
        weights = []
        for i in range(self.num_inputs):
            start_idx = i * C
            end_idx = (i + 1) * C
            weight = torch.sigmoid(y[:, start_idx:end_idx, :])
            weights.append(weight)
        
        # 加权融合 + 残差连接
        output = sum(inp * weight for inp, weight in zip(inputs, weights))
        output = output + self.residual_weight * x_combined
        
        return output

class OptimizedMovingAvg(nn.Module):
    """
    优化的移动平均 - 针对短期预测优化
    """
    def __init__(self, kernel_sizes=[8, 16, 24], stride=1):
        super(OptimizedMovingAvg, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.moving_avgs = nn.ModuleList([
            moving_avg(kernel_size, stride) for kernel_size in kernel_sizes
        ])
        # 可学习的自适应权重
        self.weights = nn.Parameter(torch.ones(len(kernel_sizes)))
        self.weight_norm = nn.LayerNorm(len(kernel_sizes))
        
    def forward(self, x):
        trends = []
        for moving_avg_layer in self.moving_avgs:
            trend = moving_avg_layer(x)
            trends.append(trend)
        
        # 归一化权重
        weights = self.weight_norm(self.weights)
        weights = F.softmax(weights, dim=0)
        combined_trend = sum(w * t for w, t in zip(weights, trends))
        return combined_trend

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    与AdaWaveNet.py完全相同
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
    与AdaWaveNet.py完全相同
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
        window_sum = torch.nn.functional.conv1d(x_padded, 
                                                weight=torch.ones((1, num_channels, self.kernel_size)).cuda(),
                                                stride=self.stride)

        # Count non-zero elements in each window
        window_count = torch.nn.functional.conv1d(non_zero_mask.float(), 
                                                  weight=torch.ones((1, num_channels, self.kernel_size)).cuda(),
                                                  stride=self.stride)

        # Avoid division by zero; set count to 1 where there are no non-zero elements
        window_count = torch.clamp(window_count, min=1)

        # Compute the moving average
        moving_avg = window_sum / window_count
        return moving_avg

class series_decomp(nn.Module):
    """
    Series decomposition block
    与AdaWaveNet.py完全相同
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

class EnhancedMultiScaleFeature(nn.Module):
    """
    增强的多尺度特征模块 - 针对96/192预测优化
    """
    def __init__(self, channels, dropout=0.1, pred_len=96):
        super().__init__()
        self.pred_len = pred_len
        
        # 根据预测长度调整卷积核大小
        if pred_len <= 96:
            kernels = [3, 5, 7]
        else:  # pred_len == 192
            kernels = [3, 7, 11]
        
        # 深度可分离卷积
        self.conv_layers = nn.ModuleList()
        for kernel in kernels:
            conv_block = nn.Sequential(
                # Depthwise convolution
                nn.Conv1d(channels, channels, kernel, padding=kernel//2, groups=channels),
                nn.BatchNorm1d(channels),
                nn.SiLU(inplace=True),
                # Pointwise convolution
                nn.Conv1d(channels, channels, 1),
                nn.BatchNorm1d(channels),
                nn.SiLU(inplace=True)
            )
            self.conv_layers.append(conv_block)
        
        # 自适应SE注意力
        self.se_attention = AdaptiveSEAttention1D(
            channel=channels, 
            reduction=max(4, channels // 8),
            num_inputs=len(kernels),
            pred_len=pred_len
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 多尺度特征提取
        features = []
        for conv_layer in self.conv_layers:
            feature = conv_layer(x)
            features.append(feature)
        
        # SE注意力融合
        enhanced = self.se_attention(*features)
        
        # 输出投影
        enhanced = self.output_proj(enhanced)
        
        # 残差连接
        output = self.dropout(enhanced) + x
        return output

class AdpWaveletBlock(nn.Module):
    """
    自适应小波块 - 与AdaWaveNet.py完全相同的小波变换逻辑
    仅在输入和输出处添加优化的SE增强
    """
    def __init__(self, configs, input_size):
        super(AdpWaveletBlock, self).__init__()
        self.regu_details = configs.regu_details
        self.regu_approx = configs.regu_approx
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        # 小波变换模块 - 与AdaWaveNet完全相同
        self.wavelet = LiftingScheme(configs.enc_in, k_size=configs.lifting_kernel_size, input_size=input_size)
        self.norm_x = normalization(configs.enc_in)
        self.norm_d = normalization(configs.enc_in)
        
        # 优化的多尺度特征增强
        self.feature_enhance = EnhancedMultiScaleFeature(
            configs.enc_in, 
            configs.dropout, 
            pred_len=getattr(configs, 'pred_len', 96)
        )

    def forward(self, x):
        # 优化的特征增强 - 在小波变换前
        x_enhanced = self.feature_enhance(x)
        
        # 小波变换 - 与AdaWaveNet完全相同的逻辑
        (c, d) = self.wavelet(x_enhanced)
        x = c

        # 正则化计算 - 与AdaWaveNet完全相同
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

        # 归一化 - 与AdaWaveNet完全相同
        x = self.norm_x(x)
        d = self.norm_d(d)
        
        return x, r, d

class InverseAdpWaveletBlock(nn.Module):
    """
    逆小波块 - 与AdaWaveNet.py完全相同
    """
    def __init__(self, configs, input_size):
        super(InverseAdpWaveletBlock, self).__init__()
        self.inverse_wavelet = InverseLiftingScheme(configs.enc_in, input_size=input_size, kernel_size=configs.lifting_kernel_size)

    def forward(self, c, d):
        reconstructed = self.inverse_wavelet(c, d)
        return reconstructed

class OptimizedClusteredLinear(nn.Module):
    """
    优化的聚类线性层 - 针对96/192预测窗口优化
    """
    def __init__(self, n_clusters, enc_in, seq_len, pred_len):
        super().__init__()
        self.n_clusters = n_clusters
        self.enc_in = enc_in
        self.pred_len = pred_len
        
        # 根据预测长度调整网络深度
        if pred_len <= 96:
            hidden_multiplier = 3
            dropout_rate = 0.1
        else:  # pred_len == 192
            hidden_multiplier = 4
            dropout_rate = 0.15
        
        # 优化的线性层
        self.linear_layers = nn.ModuleDict({
            str(cluster_id): nn.Sequential(
                nn.Linear(seq_len, seq_len * hidden_multiplier),
                nn.LayerNorm(seq_len * hidden_multiplier),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(seq_len * hidden_multiplier, seq_len * 2),
                nn.LayerNorm(seq_len * 2),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(seq_len * 2, pred_len)
            ) for cluster_id in range(n_clusters)
        })
        
        # 自适应SE注意力
        self.cluster_se = AdaptiveSEAttention1D(
            channel=enc_in,
            reduction=max(2, enc_in // 4),
            num_inputs=1,
            pred_len=pred_len
        )
        
        # 输出增强
        self.output_enhance = nn.Sequential(
            nn.Conv1d(enc_in, enc_in, 1),
            nn.BatchNorm1d(enc_in),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5)
        )
        
    def forward(self, x, clusters):
        # SE注意力增强
        x_enhanced = self.cluster_se(x)
        
        output = []
        assert self.enc_in == len(clusters)
        for channel in range(self.enc_in):
            cluster_id = str(clusters[channel].item())
            channel_data = x_enhanced[:, channel, :].unsqueeze(1)
            transformed_channel = self.linear_layers[cluster_id](channel_data)
            output.append(transformed_channel)
        output = torch.concat(output, dim=1)
        
        # 输出增强
        output = self.output_enhance(output)
        
        return output

class Model(nn.Module):
    """
    优化的SE增强AdaWaveNet模型 (YangSETD1ivate)
    专门针对96/192预测窗口优化
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
        
        # 使用标准的序列分解（与AdaWaveNet相同）
        self.series_decomp = series_decomp(imputation=self.task_name=='imputation')
        self.rev_seasonal = RevIN(configs.enc_in)
        self.rev_trend = RevIN(configs.enc_in)
        
        # 优化的聚类线性层
        self.trend_linear = OptimizedClusteredLinear(
            configs.n_clusters, configs.enc_in, self.seq_len, configs.pred_len
        )
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len // (2 ** configs.lifting_levels), 
            configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        
        # 优化的小波块
        self.encoder_levels = nn.ModuleList()
        self.linear_levels = nn.ModuleList()
        self.coef_linear_levels = nn.ModuleList()
        self.coef_dec_levels = nn.ModuleList()
        
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
            input_size = input_size // 2 
            
            # 根据预测长度优化线性变换
            if self.pred_len <= 96:
                hidden_multiplier = 3
                dropout_rate = configs.dropout
            else:  # pred_len == 192
                hidden_multiplier = 4
                dropout_rate = configs.dropout * 1.2
            
            enhanced_linear = nn.Sequential(
                nn.Linear(input_size, input_size * hidden_multiplier),
                nn.LayerNorm(input_size * hidden_multiplier),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(input_size * hidden_multiplier, input_size * expand_ratio)
            )
            
            self.linear_levels.add_module('linear_level_'+str(i), enhanced_linear)
            self.coef_linear_levels.add_module('coef_linear_level_'+str(i), enhanced_linear)
            self.coef_dec_levels.add_module('coef_dec_level_'+str(i), enhanced_linear)

        self.input_size = input_size
        
        # 解码器 - 与AdaWaveNet相同
        self.decoder_levels = nn.ModuleList()
        for i in range(configs.lifting_levels-1, -1, -1):
            self.decoder_levels.add_module(
                'decoder_level_'+str(i),
                InverseAdpWaveletBlock(configs, input_size=input_size)
            )
            input_size *= 2
        
        # 优化的低秩投影
        if self.task_name == "super_resolution":
            proj_hidden = configs.d_model * (3 if self.pred_len <= 96 else 4)
            self.lowrank_projection = nn.Sequential(
                nn.Linear(configs.d_model, proj_hidden),
                nn.LayerNorm(proj_hidden),
                nn.SiLU(inplace=True),
                nn.Dropout(configs.dropout),
                nn.Linear(proj_hidden, self.pred_len // (2 ** configs.lifting_levels))
            )
        else:
            proj_hidden = configs.d_model * (3 if self.pred_len <= 96 else 4)
            self.lowrank_projection = nn.Sequential(
                nn.Linear(configs.d_model, proj_hidden),
                nn.LayerNorm(proj_hidden),
                nn.SiLU(inplace=True),
                nn.Dropout(configs.dropout),
                nn.Linear(proj_hidden, self.seq_len // (2 ** configs.lifting_levels))
            )

        # 编码器
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
        
        # 优化的投影层
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            proj_hidden = self.seq_len * (3 if self.pred_len <= 96 else 4)
            self.projection = nn.Sequential(
                nn.Linear(self.seq_len, proj_hidden),
                nn.LayerNorm(proj_hidden),
                nn.SiLU(inplace=True),
                nn.Dropout(configs.dropout),
                nn.Linear(proj_hidden, self.seq_len),
                nn.LayerNorm(self.seq_len),
                nn.SiLU(inplace=True),
                nn.Dropout(configs.dropout * 0.5),
                nn.Linear(self.seq_len, configs.pred_len)
            )
        elif self.task_name == 'imputation':
            self.projection = nn.Sequential(
                nn.Linear(self.seq_len, self.seq_len * 3),
                nn.LayerNorm(self.seq_len * 3),
                nn.SiLU(inplace=True),
                nn.Linear(self.seq_len * 3, self.seq_len)
            )
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Sequential(
                nn.Linear(self.seq_len, self.seq_len * 3),
                nn.LayerNorm(self.seq_len * 3),
                nn.SiLU(inplace=True),
                nn.Linear(self.seq_len * 3, self.seq_len)
            )
        elif self.task_name == 'super_resolution':
            self.projection = nn.Sequential(
                nn.Linear(configs.pred_len, configs.pred_len * 3),
                nn.LayerNorm(configs.pred_len * 3),
                nn.SiLU(inplace=True),
                nn.Linear(configs.pred_len * 3, configs.pred_len)
            )

        self.register_buffer('clusters', None)
        
        # 应用权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """优化的权重初始化"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, clusters):
        # 序列分解 - 与AdaWaveNet相同
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        
        # 归一化 - 与AdaWaveNet相同
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
        
        # 编码过程 - 使用优化的小波块
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
        
        # 嵌入
        x_enc = x_enc.permute(0,2,1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # 低秩投影
        x_dec = self.lowrank_projection(enc_out)
        
        # 解码过程
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
            
        # 投影
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        
        # 反归一化
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        # 趋势预测 - 使用优化的聚类线性层
        moving_mean_out = self.trend_linear(moving_mean.permute(0,2,1), self.clusters).permute(0,2,1)
        moving_mean_out = self.rev_trend(moving_mean_out, 'denorm')
        
        # 最终输出
        dec_out = dec_out + moving_mean_out
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # 复用forecast逻辑，添加掩码处理
        base_output = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, self.clusters)
        
        if mask is not None:
            # 使用掩码信息refine输出
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
