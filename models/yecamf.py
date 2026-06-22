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

class OneConv1D(nn.Module):
    """
    1D卷积模块，适配时序数据
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                     padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class MSFBlock1D(nn.Module):
    """
    多尺度特征融合块 - 1D时序版本
    适配时序预测任务，融合不同尺度的时序特征
    """
    def __init__(self, in_channels, kernel_sizes=[3, 5, 7, 9]):
        super(MSFBlock1D, self).__init__()
        out_channels = in_channels
        self.kernel_sizes = kernel_sizes
        
        # 多尺度卷积分支
        self.multi_scale_convs = nn.ModuleList([
            OneConv1D(in_channels, out_channels, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # 通道注意力模块
        self.channel_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True)
            ) for _ in kernel_sizes
        ])
        
        # 尺度权重计算
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
        # 输出投影
        self.project = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        """
        x: (B, C, L) - 输入时序特征
        """
        B, C, L = x.shape
        residual = x
        
        # 多尺度特征提取
        multi_scale_features = []
        scale_weights = []
        
        for i, (conv, channel_att) in enumerate(zip(self.multi_scale_convs, self.channel_attentions)):
            # 多尺度卷积
            scale_feat = conv(x)  # (B, C, L)
            multi_scale_features.append(scale_feat)
            
            # 通道注意力权重计算
            scale_weight = channel_att(self.gap(scale_feat))  # (B, C, 1)
            scale_weights.append(scale_weight)
        
        # 拼接所有尺度的权重: (B, C, num_scales, 1)
        weight = torch.stack(scale_weights, dim=2)  # (B, C, num_scales, 1)
        
        # 计算尺度权重
        weight = self.softmax(self.sigmoid(weight))  # (B, C, num_scales, 1)
        
        # 加权融合多尺度特征
        fused_feature = torch.zeros_like(x)  # (B, C, L)
        for i, feat in enumerate(multi_scale_features):
            scale_weight = weight[:, :, i:i+1, :]  # (B, C, 1, 1)
            scale_weight = scale_weight.squeeze(-1)  # (B, C, 1)
            fused_feature += scale_weight * feat  # (B, C, L)
        
        # 输出投影
        output = self.project(fused_feature)
        
        # 残差连接
        output = residual * self.residual_weight + output
        
        return output

class TrendAwareAttention1D(nn.Module):
    """
    适配1D时序数据的趋势感知注意力机制
    专门用于处理趋势分量，捕获长期依赖关系
    """
    def __init__(self, d_model, n_heads=8, kernel_size=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        
        # 1D卷积用于生成query和key（捕获趋势模式）
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding)
        
        # 线性层生成value
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
        # 归一化层
        self.norm_q = nn.BatchNorm1d(d_model)
        self.norm_k = nn.BatchNorm1d(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 趋势增强模块
        self.trend_enhance = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(d_model * 2, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        """
        x: (B, L, C) - 趋势分量
        """
        B, L, C = x.shape
        residual = x
        
        # 转换为 (B, C, L) 用于1D卷积
        x_conv = x.permute(0, 2, 1)
        
        # 生成query和key（通过卷积捕获趋势模式）
        query = self.norm_q(self.conv_q(x_conv))[:, :, :-self.padding]  # (B, C, L)
        key = self.norm_k(self.conv_k(x_conv))[:, :, :-self.padding]    # (B, C, L)
        
        # 转换回 (B, L, C)
        query = query.permute(0, 2, 1)  # (B, L, C)
        key = key.permute(0, 2, 1)      # (B, L, C)
        
        # 生成value
        value = self.linear_v(x)  # (B, L, C)
        
        # 多头注意力
        query = query.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, L, d_head)
        key = key.view(B, L, self.n_heads, self.d_head).transpose(1, 2)      # (B, n_heads, L, d_head)
        value = value.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, L, d_head)
        
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, n_heads, L, L)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        attended = torch.matmul(attention_weights, value)  # (B, n_heads, L, d_head)
        attended = attended.transpose(1, 2).contiguous().view(B, L, C)  # (B, L, C)
        
        # 输出投影
        output = self.linear_out(attended)
        
        # 趋势增强
        enhanced = self.trend_enhance(output.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 残差连接
        output = residual * self.residual_weight + output + enhanced
        
        return output

class EnhancedTrendProcessor(nn.Module):
    """
    增强的趋势处理器，集成趋势感知注意力和MSF多尺度融合
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = getattr(configs, 'd_model', 512)
        
        # 输入投影
        self.input_proj = nn.Linear(self.enc_in, self.d_model)
        
        # 趋势感知注意力层
        self.trend_attention = TrendAwareAttention1D(
            d_model=self.d_model,
            n_heads=getattr(configs, 'n_heads', 8),
            kernel_size=getattr(configs, 'trend_kernel_size', 5),
            dropout=getattr(configs, 'dropout', 0.1)
        )
        
        # MSF多尺度特征融合
        self.msf_block = MSFBlock1D(
            in_channels=self.d_model,
            kernel_sizes=[3, 5, 7, 9]
        )
        
        # 趋势预测头
        self.trend_predictor = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len * 2),
            nn.GELU(),
            nn.Dropout(getattr(configs, 'dropout', 0.1)),
            nn.Linear(self.pred_len * 2, self.pred_len)
        )
        
        # 输出投影
        self.output_proj = nn.Linear(self.d_model, self.enc_in)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.d_model)
        
    def forward(self, trend_input):
        """
        trend_input: (B, L, C) - 趋势分量
        返回: (B, pred_len, C) - 预测的趋势分量
        """
        B, L, C = trend_input.shape
        
        # 输入投影到高维空间
        x = self.input_proj(trend_input)  # (B, L, d_model)
        x = self.layer_norm(x)
        
        # 趋势感知注意力
        x_attended = self.trend_attention(x)  # (B, L, d_model)
        
        # MSF多尺度特征融合
        x_conv = x_attended.permute(0, 2, 1)  # (B, d_model, L)
        x_fused = self.msf_block(x_conv)  # (B, d_model, L)
        x_fused = x_fused.permute(0, 2, 1)  # (B, L, d_model)
        
        # 残差连接和归一化
        enhanced_features = x_attended + x_fused
        enhanced_features = self.layer_norm(enhanced_features)
        
        # 趋势预测
        pred_features = self.trend_predictor(enhanced_features.transpose(1, 2)).transpose(1, 2)  # (B, pred_len, d_model)
        
        # 输出投影回原始维度
        trend_output = self.output_proj(pred_features)  # (B, pred_len, C)
        
        return trend_output

class ECAAttention1D(nn.Module):
    """
    优化版ECA注意力（1D）
    - 支持Avg/Max双分支并融合
    - 自适应卷积核大小（可选）
    - 温度缩放提升稳定性
    """
    def __init__(self, channel=512, kernel_size=3, dropout=0.1, use_max=True, adaptive_kernel=False):
        super().__init__()
        self.channel = channel
        self.use_max = use_max
        if adaptive_kernel:
            k = int(abs((math.log2(channel) / 2)))
            k = k if k % 2 else k + 1
            kernel_size = max(3, min(9, k))
        self.kernel_size = kernel_size
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1) if use_max else None
        
        self.conv_avg = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.conv_max = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False) if use_max else None
        
        self.bn_avg = nn.BatchNorm1d(1)
        self.bn_max = nn.BatchNorm1d(1) if use_max else None
        self.dropout = nn.Dropout(dropout)
        
        # 融合权重（学习Avg/Max的重要性）
        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5])) if use_max else None
        
        self.sigmoid = nn.Sigmoid()
        self.temperature = nn.Parameter(torch.ones(1))
        self.scale = nn.Parameter(torch.ones(1))
        
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
        init.constant_(self.scale, 0.1)
        init.constant_(self.temperature, 1.0)

    def forward(self, x):
        B, C, L = x.size()
        residual = x
        
        # Avg path
        y_avg = self.gap(x)                # (B, C, 1)
        y_avg = y_avg.permute(0, 2, 1)     # (B, 1, C)
        y_avg = self.bn_avg(self.conv_avg(y_avg))
        y_avg = self.dropout(y_avg)
        
        if self.use_max:
            y_max = self.gmp(x)            # (B, C, 1)
            y_max = y_max.permute(0, 2, 1) # (B, 1, C)
            y_max = self.bn_max(self.conv_max(y_max))
            y_max = self.dropout(y_max)
            
            weights = torch.softmax(self.fusion_weight, dim=0)
            y = weights[0] * y_avg + weights[1] * y_max
        else:
            y = y_avg
        
        # Temperature-scaled sigmoid
        temp = torch.clamp(self.temperature, 0.5, 5.0)
        y = self.sigmoid(y / temp)
        
        # (B, 1, C) -> (B, C, 1) -> broadcast to (B, C, L)
        y = y.permute(0, 2, 1)
        attention_weights = y.expand(-1, -1, L)
        
        enhanced_x = x * attention_weights
        output = residual + enhanced_x * self.scale
        return output

class ECABlock1D(nn.Module):
    """
    集成MSF多尺度融合的ECA块
    """
    def __init__(self, channel=512, kernel_size=3, dropout=0.1, use_max=True, adaptive_kernel=False):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channel)
        
        # ECA注意力
        self.eca_attention = ECAAttention1D(
            channel=channel, kernel_size=kernel_size, 
            dropout=dropout, use_max=use_max, adaptive_kernel=adaptive_kernel
        )
        
        # MSF多尺度特征融合
        self.msf_block = MSFBlock1D(
            in_channels=channel,
            kernel_sizes=[3, 5, 7]  # 适中的尺度数量
        )
        
        # 特征增强
        self.feature_enhance = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(channel),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channel, channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(channel)
        )
        
        self.residual_scale = nn.Parameter(torch.ones(1))
        self.attention_scale = nn.Parameter(torch.ones(1))
        self.msf_scale = nn.Parameter(torch.ones(1))
        
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
        init.constant_(self.residual_scale, 0.1)
        init.constant_(self.attention_scale, 0.1)
        init.constant_(self.msf_scale, 0.1)

    def forward(self, x):
        residual = x
        
        # 预归一化
        x_norm = self.pre_norm(x.transpose(1, 2)).transpose(1, 2)
        
        # ECA注意力
        attention_out = self.eca_attention(x_norm)
        
        # MSF多尺度特征融合
        msf_out = self.msf_block(attention_out)
        
        # 特征增强
        enhanced_out = self.feature_enhance(msf_out)
        
        # 多分支融合
        output = (residual * self.residual_scale + 
                 attention_out * self.attention_scale + 
                 enhanced_out * self.msf_scale)
        
        return output

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

class moving_avg_imputation(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg_imputation, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        num_channels = x.shape[1]
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2))
        end = x[:, :, -1:].repeat(1, 1, math.floor((self.kernel_size - 1) // 2))
        x_padded = torch.cat([front, x, end], dim=-1)
        non_zero_mask = x_padded != 0
        weight = torch.ones((1, num_channels, self.kernel_size), device=x_padded.device)
        window_sum = torch.nn.functional.conv1d(x_padded, weight=weight, stride=self.stride)
        window_count = torch.nn.functional.conv1d(non_zero_mask.float(), weight=weight, stride=self.stride)
        window_count = torch.clamp(window_count, min=1)
        moving_avg = window_sum / window_count
        return moving_avg

class series_decomp(nn.Module):
    def __init__(self, kernel_size=24, stride=1, imputation=False):
        super(series_decomp, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.moving_avg = moving_avg(kernel_size, stride=stride) if not imputation else moving_avg_imputation(self.kernel_size, self.stride)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class AdpWaveletBlock(nn.Module):
    def __init__(self, configs, input_size):
        super(AdpWaveletBlock, self).__init__()
        self.regu_details = getattr(configs, 'regu_details', 0.0)
        self.regu_approx = getattr(configs, 'regu_approx', 0.0)
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

class Model(nn.Module):
    """
    YECAMF: YECA2 + MSF多尺度特征融合
    集成了趋势感知注意力、ECA注意力和MSF多尺度特征融合
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'super_resolution':
            self.seq_len = self.seq_len // configs.sr_ratio
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # 序列分解
        self.series_decomp = series_decomp(imputation = self.task_name=='imputation')
        self.rev_seasonal = RevIN(configs.enc_in)
        self.rev_trend = RevIN(configs.enc_in)

        # 使用增强的趋势处理器（集成趋势感知注意力和MSF）
        self.trend_processor = EnhancedTrendProcessor(configs)

        # ECA块（集成MSF多尺度融合）
        eca_kernel_size = getattr(configs, 'eca_kernel_size', 3)
        eca_dropout = getattr(configs, 'eca_dropout', None)
        if eca_dropout is None:
            eca_dropout = configs.dropout
        eca_use_max = getattr(configs, 'eca_use_max', True)
        eca_adaptive_kernel = getattr(configs, 'eca_adaptive_kernel', False)
        self.eca_block = ECABlock1D(
            channel=configs.enc_in,
            kernel_size=eca_kernel_size,
            dropout=eca_dropout,
            use_max=eca_use_max,
            adaptive_kernel=eca_adaptive_kernel
        )

        self.enc_embedding = DataEmbedding_inverted(self.seq_len // (2 ** configs.lifting_levels), configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.encoder_levels = nn.ModuleList()
        self.linear_levels = nn.ModuleList()
        self.coef_linear_levels = nn.ModuleList()
        self.coef_dec_levels = nn.ModuleList()
        in_planes = configs.enc_in
        input_size = self.seq_len
        expand_ratio = configs.sr_ratio if self.task_name == "super_resolution" else 1
        for i in range(configs.lifting_levels):
            self.encoder_levels.add_module('encoder_level_'+str(i), AdpWaveletBlock(configs, input_size))
            in_planes *= 1
            input_size = input_size // 2
            self.linear_levels.add_module('linear_level_'+str(i), nn.Sequential(nn.Linear(input_size, input_size * expand_ratio)))
            self.coef_linear_levels.add_module('linear_level_'+str(i), nn.Sequential(nn.Linear(input_size, input_size * expand_ratio)))
            self.coef_dec_levels.add_module('linear_level_'+str(i), nn.Sequential(nn.Linear(input_size, input_size * expand_ratio)))

        self.input_size = input_size

        self.decoder_levels = nn.ModuleList()
        for i in range(configs.lifting_levels-1, -1, -1):
            self.decoder_levels.add_module('decoder_level_'+str(i), InverseAdpWaveletBlock(configs, input_size=input_size))
            in_planes //= 1
            input_size *= 2

        if self.task_name == "super_resolution":
            self.lowrank_projection = nn.Linear(configs.d_model, self.pred_len // (2 ** configs.lifting_levels), bias=True)
        else:
            self.lowrank_projection = nn.Linear(configs.d_model, self.seq_len // (2 ** configs.lifting_levels), bias=True)

        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for l in range(configs.e_layers)
        ], norm_layer=torch.nn.LayerNorm(configs.d_model))

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(self.seq_len, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(self.seq_len, self.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(self.seq_len, self.seq_len, bias=True)
        if self.task_name == 'super_resolution':
            self.projection = nn.Linear(configs.pred_len, configs.pred_len, bias=True)

    def _norm_enc(self, x):
        x_enc = x.permute(0,2,1)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        return x_enc, means, stdev

    def _denorm(self, dec_out, means, stdev):
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def _common_decode(self, x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels):
        x_enc = x_enc.permute(0,2,1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        for dec, x_emb_level, coef_emb_level, c_linear in zip(self.decoder_levels, x_embedding_levels[::-1], coef_embedding_levels[::-1], self.coef_dec_levels[::-1]):
            details = encoded_coefficients.pop()
            details = coef_emb_level + c_linear(details)
            x_dec = x_dec + x_emb_level
            x_dec = dec(x_dec, details)
        return x_dec

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 序列分解
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        
        # ECA + MSF增强的季节性处理
        x = self.eca_block(x)
        x_enc, means, stdev = self._norm_enc(x)
        _, _, N = x_enc.shape
        moving_mean = self.rev_trend(moving_mean, 'norm')
        
        # 小波编码
        x_enc = x_enc.permute(0,2,1)
        encoded_coefficients, x_embedding_levels, coef_embedding_levels = [], [], []
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
        
        # 小波解码
        x_dec = self._common_decode(x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denorm(dec_out, means, stdev)
        
        # 使用增强的趋势处理器（集成趋势感知注意力和MSF）
        moving_mean_out = self.trend_processor(moving_mean)
        moving_mean_out = self.rev_trend(moving_mean_out, 'denorm')
        
        # 最终融合
        dec_out = dec_out + moving_mean_out
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        x = self.eca_block(x)
        x_enc, means, stdev = self._norm_enc(x)
        _, _, N = x_enc.shape
        moving_mean = self.rev_trend(moving_mean, 'norm')
        x_enc = x_enc.permute(0,2,1)
        encoded_coefficients, x_embedding_levels, coef_embedding_levels = [], [], []
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
        x_dec = self._common_decode(x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denorm(dec_out, means, stdev)
        
        moving_mean_out = self.trend_processor(moving_mean)
        moving_mean_out = self.rev_trend(moving_mean_out, 'denorm')
        dec_out = dec_out + moving_mean_out
        return dec_out

    def anomaly_detection(self, x_enc):
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        x = self.eca_block(x)
        x_enc, means, stdev = self._norm_enc(x)
        _, _, N = x_enc.shape
        moving_mean = self.rev_trend(moving_mean, 'norm')
        x_enc = x_enc.permute(0,2,1)
        encoded_coefficients, x_embedding_levels, coef_embedding_levels = [], [], []
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
        x_dec = self._common_decode(x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denorm(dec_out, means, stdev)
        
        moving_mean_out = self.trend_processor(moving_mean)
        moving_mean_out = self.rev_trend(moving_mean_out, 'denorm')
        dec_out = dec_out + moving_mean_out
        return dec_out

    def super_resolution(self, x_enc):
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        x = self.eca_block(x)
        x_enc, means, stdev = self._norm_enc(x)
        _, _, N = x_enc.shape
        moving_mean = self.rev_trend(moving_mean, 'norm')
        x_enc = x_enc.permute(0,2,1)
        encoded_coefficients, x_embedding_levels, coef_embedding_levels = [], [], []
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
        x_dec = self._common_decode(x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denorm(dec_out, means, stdev)
        
        moving_mean_out = self.trend_processor(moving_mean)
        moving_mean_out = self.rev_trend(moving_mean_out, 'denorm')
        dec_out = dec_out + moving_mean_out
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            raise ValueError('classification not implemented for yecamf')
        if self.task_name == 'super_resolution':
            dec_out = self.super_resolution(x_enc)
            return dec_out
        return None
