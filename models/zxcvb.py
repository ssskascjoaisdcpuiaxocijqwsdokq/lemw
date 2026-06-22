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

class SCSA1D_MAE_MSE_Optimized(nn.Module):
    """
    1D适配的SCSA空间通道协同注意力模块
    专门针对时序预测的MAE/MSE优化
    """
    def __init__(
        self,
        dim: int,
        head_num: int = 8,
        group_kernel_sizes: list = [3, 5, 7, 9],
        qkv_bias: bool = False,
        attn_drop_ratio: float = 0.1,
        gate_layer: str = 'sigmoid',
    ):
        super(SCSA1D_MAE_MSE_Optimized, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        
        # 确保维度可被4整除
        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = self.dim // 4
        
        # 多尺度空间注意力 - 1D适配
        # 优化1: 使用不同kernel size捕获多尺度时序特征
        self.local_dwc = nn.Conv1d(
            self.group_chans, self.group_chans, 
            kernel_size=group_kernel_sizes[0],
            padding=group_kernel_sizes[0] // 2, 
            groups=self.group_chans
        )
        self.global_dwc_s = nn.Conv1d(
            self.group_chans, self.group_chans,
            kernel_size=group_kernel_sizes[1],
            padding=group_kernel_sizes[1] // 2,
            groups=self.group_chans
        )
        self.global_dwc_m = nn.Conv1d(
            self.group_chans, self.group_chans,
            kernel_size=group_kernel_sizes[2],
            padding=group_kernel_sizes[2] // 2,
            groups=self.group_chans
        )
        self.global_dwc_l = nn.Conv1d(
            self.group_chans, self.group_chans,
            kernel_size=group_kernel_sizes[3],
            padding=group_kernel_sizes[3] // 2,
            groups=self.group_chans
        )
        
        # 优化2: 使用Sigmoid门控，更适合回归任务
        self.sa_gate = nn.Sigmoid() if gate_layer == 'sigmoid' else nn.Softmax(dim=1)
        
        # 优化3: 使用LayerNorm替代GroupNorm，更适合时序数据
        self.norm_spatial = nn.LayerNorm(dim)
        
        # 渐进式通道自注意力 - 1D适配
        self.norm_channel = nn.LayerNorm(dim)
        
        # 优化4: 使用1D卷积替代2D卷积
        self.q = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Sigmoid() if gate_layer == 'sigmoid' else nn.Softmax(dim=1)
        
        # 优化5: 下采样用于减少计算复杂度
        self.down_sample = nn.AdaptiveAvgPool1d(1)
        
        # 优化6: 特征融合增强层
        self.feature_enhance = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x (B, L, C) - 时序数据
        输出: enhanced_x (B, L, C) - 增强后的时序数据
        """
        B, L, C = x.shape
        
        # 转换为卷积格式: (B, C, L)
        x_conv = x.permute(0, 2, 1)
        
        # === 多尺度空间注意力处理 ===
        # 优化7: 分组处理不同尺度的时序特征
        l_x, g_x_s, g_x_m, g_x_l = torch.split(x_conv, self.group_chans, dim=1)
        
        # 多尺度特征提取
        local_feat = self.local_dwc(l_x)          # 局部特征 (短期依赖)
        global_feat_s = self.global_dwc_s(g_x_s)  # 小尺度全局特征
        global_feat_m = self.global_dwc_m(g_x_m)  # 中尺度全局特征  
        global_feat_l = self.global_dwc_l(g_x_l)  # 大尺度全局特征 (长期依赖)
        
        # 拼接多尺度特征
        multi_scale_feat = torch.cat([
            local_feat, global_feat_s, global_feat_m, global_feat_l
        ], dim=1)  # (B, C, L)
        
        # 空间注意力权重生成
        spatial_attn = self.sa_gate(multi_scale_feat)
        
        # 应用空间注意力
        x_spatial = x_conv * spatial_attn
        
        # === 渐进式通道自注意力处理 ===
        # 优化8: 下采样减少计算量，专注于全局通道关系
        y = self.down_sample(x_spatial)  # (B, C, 1)
        
        # 生成Q, K, V
        q = self.q(y)  # (B, C, 1)
        k = self.k(y)  # (B, C, 1)
        v = self.v(y)  # (B, C, 1)
        
        # 重塑为多头注意力格式
        q = q.view(B, self.head_num, self.head_dim, 1)
        k = k.view(B, self.head_num, self.head_dim, 1)
        v = v.view(B, self.head_num, self.head_dim, 1)
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scaler  # (B, head_num, head_dim, head_dim)
        attn = self.attn_drop(attn.softmax(dim=-1))
        
        # 应用注意力到V
        attn_out = attn @ v  # (B, head_num, head_dim, 1)
        
        # 重塑回原始格式
        attn_out = attn_out.view(B, C, 1)  # (B, C, 1)
        
        # 通道注意力权重
        channel_attn = self.ca_gate(attn_out)  # (B, C, 1)
        
        # 应用通道注意力
        x_channel = x_spatial * channel_attn  # 广播到 (B, C, L)
        
        # === 特征增强和融合 ===
        # 优化9: 残差连接 + 特征增强
        enhanced_feat = self.feature_enhance(x_channel)
        
        # 优化10: 多层次残差连接
        output = x_conv + x_spatial + enhanced_feat
        
        # 转换回时序格式: (B, L, C)
        output = output.permute(0, 2, 1)
        
        # 优化11: 最终归一化
        output = self.norm_spatial(output)
        
        return output

class SCSAEnhancedEncoder(nn.Module):
    """
    基于SCSA的增强编码器，专门优化MAE/MSE
    """
    def __init__(self, configs):
        super(SCSAEnhancedEncoder, self).__init__()
        
        # 多个SCSA层用于深度特征提取
        self.scsa_layers = nn.ModuleList([
            SCSA1D_MAE_MSE_Optimized(
                dim=configs.d_model,
                head_num=configs.n_heads,
                group_kernel_sizes=[3, 5, 7, 9],
                attn_drop_ratio=configs.dropout * 0.5,  # 减少dropout
            ),
            SCSA1D_MAE_MSE_Optimized(
                dim=configs.d_model,
                head_num=configs.n_heads,
                group_kernel_sizes=[5, 7, 9, 11],  # 更大的感受野
                attn_drop_ratio=configs.dropout * 0.3,
            )
        ])
        
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
        
        # 优化12: 自适应特征融合权重
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)  # SCSA1, SCSA2, Transformer
        
        # 优化13: MAE/MSE专用预测增强层
        self.prediction_enhance = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model * 2),
            nn.GELU(),
            nn.Dropout(configs.dropout * 0.2),
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.LayerNorm(configs.d_model),
        )
        
        self.final_norm = nn.LayerNorm(configs.d_model)
        
    def forward(self, x, attn_mask=None):
        # x: (B, L, C)
        
        # 多层SCSA处理
        scsa_outputs = []
        current_x = x
        
        for scsa_layer in self.scsa_layers:
            scsa_out = scsa_layer(current_x)
            scsa_outputs.append(scsa_out)
            # 为下一层准备输入（添加残差连接）
            current_x = current_x + scsa_out
        
        # 标准Transformer处理
        transformer_out, attns = self.transformer_encoder(x, attn_mask=attn_mask)
        
        # 自适应融合
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_output = (weights[0] * scsa_outputs[0] + 
                       weights[1] * scsa_outputs[1] + 
                       weights[2] * transformer_out)
        
        # 预测增强
        enhanced_output = self.prediction_enhance(fused_output)
        
        # 最终输出（多重残差连接）
        output = self.final_norm(x + fused_output + enhanced_output)
        
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
    ZXCVB: AdaWaveNet + SCSA1D_MAE_MSE_Optimized
    融合空间通道协同注意力的时序预测模型
    
    主要创新:
    1. 多尺度时序特征提取 (3,5,7,9 kernel sizes)
    2. 空间-通道协同注意力机制
    3. 渐进式通道自注意力
    4. 多层次残差连接
    5. MAE/MSE专用优化策略
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

        # SCSA增强编码器
        self.encoder = SCSAEnhancedEncoder(configs)
        
        # 优化14: 高精度预测头
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Sequential(
                nn.Linear(self.seq_len, configs.pred_len * 2),
                nn.ReLU(inplace=True),
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
        
        # SCSA增强编码器处理
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



















