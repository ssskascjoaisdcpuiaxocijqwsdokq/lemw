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

class BasicConv1D(nn.Module):
    """1D基础卷积模块"""
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True):
        super(BasicConv1D, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        layers.append(nn.Conv1d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        
        if norm:
            layers.append(nn.BatchNorm1d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class DynamicFilter1D(nn.Module):
    """1D动态滤波器 - 简化版本，避免复杂unfold操作"""
    def __init__(self, inchannels, kernel_size=3, dilation=1, stride=1, group=8):
        super(DynamicFilter1D, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # 简化为标准卷积操作
        self.dynamic_conv = nn.Conv1d(inchannels, inchannels, kernel_size, 
                                     padding=dilation * (kernel_size - 1) // 2, 
                                     dilation=dilation, groups=min(group, inchannels), bias=False)
        self.bn = nn.BatchNorm1d(inchannels)
        self.act = nn.Tanh()
        
        # 权重生成器
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(inchannels, inchannels // 4, 1),
            nn.ReLU(),
            nn.Conv1d(inchannels // 4, inchannels, 1),
            nn.Sigmoid()
        )
        
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)

    def forward(self, x):
        identity_input = x
        
        # 动态卷积处理
        dynamic_out = self.dynamic_conv(x)
        dynamic_out = self.bn(dynamic_out)
        dynamic_out = self.act(dynamic_out)
        
        # 生成自适应权重
        weights = self.weight_gen(x)
        
        # 加权融合
        out_low = dynamic_out * weights * self.lamb_l[None, :, None]
        out_high = identity_input * (self.lamb_h[None, :, None] + 1.)

        return out_low + out_high

class SpatialStripAtt1D(nn.Module):
    """1D空间条带注意力 - 简化版本"""
    def __init__(self, dim, kernel=3, dilation=1, group=2):
        super(SpatialStripAtt1D, self).__init__()
        
        self.k = kernel
        self.dilation = dilation
        
        # 简化为标准卷积注意力
        self.spatial_conv = nn.Conv1d(dim, dim, kernel, 
                                     padding=dilation * (kernel - 1) // 2, 
                                     dilation=dilation, groups=min(group, dim), bias=False)
        
        # 注意力权重生成
        self.attention_gen = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv1d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        
        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        identity_input = x.clone()
        
        # 空间卷积处理
        spatial_out = self.spatial_conv(x)
        spatial_out = self.norm(spatial_out)
        
        # 生成注意力权重
        attention_weights = self.attention_gen(x)
        
        # 加权融合
        out_low = spatial_out * attention_weights * self.lamb_l[None, :, None]
        out_high = identity_input * (self.lamb_h[None, :, None] + 1.)

        return out_low + out_high

class CubicAttention1D(nn.Module):
    """1D立体注意力机制"""
    def __init__(self, dim, group, dilation, kernel):
        super(CubicAttention1D, self).__init__()
        
        # 1D只需要一个方向的条带注意力
        self.spatial_att = SpatialStripAtt1D(dim, dilation=dilation, group=group, kernel=kernel)
        self.gamma = nn.Parameter(torch.zeros(dim, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x):
        out = self.spatial_att(x)
        return self.gamma * out + x * self.beta

class MultiShapeKernel1D(nn.Module):
    """1D多形状核模块"""
    def __init__(self, dim, kernel_size=3, dilation=1, group=8):
        super(MultiShapeKernel1D, self).__init__()
        
        self.dynamic_filter = DynamicFilter1D(inchannels=dim, dilation=dilation, group=group, kernel_size=kernel_size)
        self.cubic_att = CubicAttention1D(dim, group=group, dilation=dilation, kernel=kernel_size)

    def forward(self, x):
        x1 = self.cubic_att(x)
        x2 = self.dynamic_filter(x)
        return x1 + x2

class MSM1D(nn.Module):
    """
    1D多尺度特征提取模块 - 适配时序预测
    基于TPAMI 2024 MSM模块，针对时序数据优化
    """
    def __init__(self, k, k_out):
        super(MSM1D, self).__init__()
        # 优化1: 适配1D的池化尺寸 - 减小下采样率避免尺寸过小
        self.pools_sizes = [4, 2, 1]  # 时序下采样率，避免输出尺寸过小
        dilation = [2, 3, 5]  # 减小膨胀率，适合时序数据
        
        pools, convs, dynas = [], [], []
        for j, i in enumerate(self.pools_sizes):
            pools.append(nn.AvgPool1d(kernel_size=i, stride=i))
            convs.append(nn.Conv1d(k, k, 3, 1, 1, bias=False))
            dynas.append(MultiShapeKernel1D(dim=k, kernel_size=3, dilation=dilation[j]))
            
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv1d(k, k_out, 3, 1, 1, bias=False)

    def forward(self, x):
        # x: (B, C, L)
        x_size = x.size()
        resl = x
        y_up = None
        
        for i in range(len(self.pools_sizes)):
            # 处理当前尺度
            if self.pools_sizes[i] == 1:  # 避免1x1池化
                current_x = x
            else:
                current_x = self.pools[i](x)
            
            # 如果有上一层的输出，需要调整尺寸匹配
            if i > 0 and y_up is not None:
                if y_up.size(2) != current_x.size(2):
                    y_up = F.interpolate(y_up, size=current_x.size(2), mode='linear', align_corners=True)
                current_x = current_x + y_up
            
            # 卷积和动态处理
            conv_out = self.convs[i](current_x)
            y = self.dynas[i](conv_out)
                
            # 上采样到原始尺寸并累加
            if y.size(2) != x_size[2]:
                y_resized = F.interpolate(y, size=x_size[2], mode='linear', align_corners=True)
            else:
                y_resized = y
            resl = torch.add(resl, y_resized)
            
            # 为下一层准备上采样输出
            if i != len(self.pools_sizes) - 1:
                y_up = y
                
        resl = self.relu(resl)
        resl = self.conv_sum(resl)
        
        return resl

class ResBlock1D(nn.Module):
    """1D残差块，集成MSM"""
    def __init__(self, in_channel, out_channel, use_msm=True):
        super(ResBlock1D, self).__init__()
        self.main = nn.Sequential(
            BasicConv1D(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            MSM1D(in_channel, out_channel) if use_msm else nn.Identity(),
            BasicConv1D(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        
        # 通道数不匹配时的投影层
        self.shortcut = nn.Identity() if in_channel == out_channel else nn.Conv1d(in_channel, out_channel, 1)

    def forward(self, x):
        return self.main(x) + self.shortcut(x)

class MSMEnhancedEncoder(nn.Module):
    """
    基于MSM的增强编码器
    """
    def __init__(self, configs):
        super(MSMEnhancedEncoder, self).__init__()
        
        # 优化2: 多级MSM处理
        self.msm_layers = nn.ModuleList([
            MSM1D(configs.d_model, configs.d_model),
            MSM1D(configs.d_model, configs.d_model),
        ])
        
        # 优化3: 残差块增强
        self.res_blocks = nn.ModuleList([
            ResBlock1D(configs.d_model, configs.d_model, use_msm=True),
            ResBlock1D(configs.d_model, configs.d_model, use_msm=False),  # 交替使用
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
        
        # 优化4: 多尺度特征融合
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv1d(configs.d_model * 3, configs.d_model, 1),  # MSM1 + MSM2 + Transformer
            nn.BatchNorm1d(configs.d_model),
            nn.GELU(),
            nn.Conv1d(configs.d_model, configs.d_model, 3, padding=1),
        )
        
        # 优化5: 自适应权重融合
        self.adaptive_weights = nn.Parameter(torch.ones(3) / 3)  # MSM1, MSM2, Transformer
        
        self.final_norm = nn.LayerNorm(configs.d_model)
        
    def forward(self, x, attn_mask=None):
        # x: (B, L, C)
        B, L, C = x.shape
        
        # 转换为卷积格式
        x_conv = x.permute(0, 2, 1)  # (B, C, L)
        
        # 多级MSM处理
        msm_outputs = []
        current_x = x_conv
        
        for i, (msm_layer, res_block) in enumerate(zip(self.msm_layers, self.res_blocks)):
            msm_out = msm_layer(current_x)
            res_out = res_block(current_x)
            
            # 融合MSM和残差输出
            fused = msm_out + res_out
            msm_outputs.append(fused)
            
            # 为下一层准备输入
            current_x = fused
        
        # 标准Transformer处理
        transformer_out, attns = self.transformer_encoder(x, attn_mask=attn_mask)
        transformer_conv = transformer_out.permute(0, 2, 1)  # (B, C, L)
        
        # 自适应权重融合
        weights = F.softmax(self.adaptive_weights, dim=0)
        
        # 加权融合三个分支
        fused_conv = (weights[0] * msm_outputs[0] + 
                     weights[1] * msm_outputs[1] + 
                     weights[2] * transformer_conv)
        
        # 多尺度特征融合
        all_features = torch.cat([msm_outputs[0], msm_outputs[1], transformer_conv], dim=1)
        enhanced_features = self.multi_scale_fusion(all_features)
        
        # 最终融合
        final_conv = fused_conv + enhanced_features
        
        # 转换回时序格式
        output = final_conv.permute(0, 2, 1)  # (B, L, C)
        
        # 残差连接和归一化
        output = self.final_norm(x + output)
        
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
    ZMEM: AdaWaveNet + MSM1D Multi-Scale Feature Extraction
    融合TPAMI 2024 MSM多尺度特征提取的时序预测模型
    
    主要创新:
    1. 多分支多尺度特征提取 (8,4,2倍下采样)
    2. 动态滤波和立体注意力机制
    3. 多级信息融合和特征还原
    4. 自适应权重融合策略
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

        # MSM增强编码器
        self.encoder = MSMEnhancedEncoder(configs)
        
        # 优化6: MSM增强的预测头
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Sequential(
                nn.Linear(self.seq_len, configs.pred_len * 2),
                nn.GELU(),
                nn.Dropout(configs.dropout * 0.3),
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
        
        # MSM增强编码器处理
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
