import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fast_pytorch_kmeans import KMeans
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.LiftingScheme import LiftingScheme, InverseLiftingScheme
from layers.Invertible import RevIN

def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)

class MSFblock_TS(nn.Module):
    """多尺度特征融合块 - 适配时序数据"""
    def __init__(self, in_channels):
        super(MSFblock_TS, self).__init__()
        out_channels = in_channels

        self.project = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
        # 四个不同尺度的SE模块
        self.SE1 = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.SE2 = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.SE3 = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.SE4 = nn.Conv1d(in_channels, in_channels, 1, bias=False)

    def forward(self, x0, x1, x2, x3):
        # x0/x1/x2/x3: (B,C,T) - 四个不同尺度的特征
        y0, y1, y2, y3 = x0, x1, x2, x3

        # 通过全局平均池化聚合信息，然后通过1x1卷积建模通道相关性
        y0_weight = self.SE1(self.gap(x0))  # (B,C,T) -> (B,C,1) -> (B,C,1)
        y1_weight = self.SE2(self.gap(x1))
        y2_weight = self.SE3(self.gap(x2))
        y3_weight = self.SE4(self.gap(x3))

        # 将多个尺度的全局信息进行拼接: (B,C,4)
        weight = torch.cat([y0_weight, y1_weight, y2_weight, y3_weight], 2)
        # 通过sigmoid和softmax获得每个尺度的权重
        weight = self.softmax(self.sigmoid(weight))

        # 提取每个尺度的权重
        y0_weight = weight[:, :, 0:1]  # (B,C,1)
        y1_weight = weight[:, :, 1:2]
        y2_weight = weight[:, :, 2:3]
        y3_weight = weight[:, :, 3:4]

        # 加权融合多尺度特征
        x_att = y0_weight * y0 + y1_weight * y1 + y2_weight * y2 + y3_weight * y3
        return self.project(x_att)

class SpatialAttention_TS(nn.Module):
    """空间注意力 - 适配时序数据"""
    def __init__(self):
        super(SpatialAttention_TS, self).__init__()
        self.sa = nn.Conv1d(2, 1, 7, padding=3, bias=True)

    def forward(self, x):
        # x: (B,C,T)
        x_avg = torch.mean(x, dim=1, keepdim=True)  # (B,1,T)
        x_max, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,T)
        x2 = torch.cat([x_avg, x_max], dim=1)  # (B,2,T)
        sattn = self.sa(x2)  # (B,1,T)
        return sattn

class ChannelAttention_TS(nn.Module):
    """通道注意力 - 适配时序数据"""
    def __init__(self, dim, reduction=8):
        super(ChannelAttention_TS, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        # 确保减少后的维度至少为1
        reduced_dim = max(1, dim // reduction)
        self.ca = nn.Sequential(
            nn.Conv1d(dim, reduced_dim, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_dim, dim, 1, bias=True),
        )

    def forward(self, x):
        # x: (B,C,T)
        x_gap = self.gap(x)  # (B,C,1)
        cattn = self.ca(x_gap)  # (B,C,1)
        return cattn

class PixelAttention_TS(nn.Module):
    """像素注意力 - 适配时序数据"""
    def __init__(self, dim):
        super(PixelAttention_TS, self).__init__()
        self.pa2 = nn.Conv1d(2 * dim, dim, 7, padding=3, groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        # x: (B,C,T), pattn1: (B,C,T)
        B, C, T = x.shape
        x = x.unsqueeze(dim=2)  # (B,C,1,T)
        pattn1 = pattn1.unsqueeze(dim=2)  # (B,C,1,T)
        x2 = torch.cat([x, pattn1], dim=2)  # (B,C,2,T)
        x2 = x2.view(B, 2*C, T)  # (B,2C,T)
        pattn2 = self.pa2(x2)  # (B,C,T)
        return pattn2

class CGAFusion_TS(nn.Module):
    """CGA融合模块 - 适配时序数据"""
    def __init__(self, dim, reduction=8):
        super(CGAFusion_TS, self).__init__()
        self.sa = SpatialAttention_TS()
        self.ca = ChannelAttention_TS(dim, reduction)
        self.pa = PixelAttention_TS(dim)
        self.conv = nn.Conv1d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # x, y: (B,C,T)
        initial = x + y
        cattn = self.ca(initial)  # (B,C,1)
        sattn = self.sa(initial)  # (B,1,T)
        
        # 广播融合通道和空间注意力
        pattn1 = sattn + cattn  # (B,C,T)
        pattn2 = self.sigmoid(self.pa(initial, pattn1))  # (B,C,T)
        
        # 加权融合
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

class moving_avg(nn.Module):
    """移动平均块"""
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2))
        end = x[:, :, -1:].repeat(1, 1, math.floor((self.kernel_size - 1) // 2))
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x

class series_decomp(nn.Module):
    """序列分解块"""
    def __init__(self, kernel_size=24, stride=1):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=stride)

    def forward(self, x):
        moving_mean = self.moving_avg(x) 
        res = x - moving_mean
        return res, moving_mean

class EnhancedAdpWaveletBlock_MSF_DEA(nn.Module):
    """增强的自适应小波块 - 集成MSF和DEA注意力"""
    def __init__(self, configs, input_size):
        super(EnhancedAdpWaveletBlock_MSF_DEA, self).__init__()
        self.regu_details = configs.regu_details
        self.regu_approx = configs.regu_approx
        
        # 保持原有的小波结构不变
        self.wavelet = LiftingScheme(configs.enc_in, k_size=configs.lifting_kernel_size, input_size=input_size)
        self.norm_x = normalization(configs.enc_in)
        self.norm_d = normalization(configs.enc_in)
        
        # 多尺度卷积生成不同尺度特征
        self.conv3 = nn.Conv1d(configs.enc_in, configs.enc_in, 3, padding=1, groups=configs.enc_in)
        self.conv5 = nn.Conv1d(configs.enc_in, configs.enc_in, 5, padding=2, groups=configs.enc_in)
        self.conv7 = nn.Conv1d(configs.enc_in, configs.enc_in, 7, padding=3, groups=configs.enc_in)
        self.conv9 = nn.Conv1d(configs.enc_in, configs.enc_in, 9, padding=4, groups=configs.enc_in)
        
        # MSF多尺度融合
        self.msf_fusion = MSFblock_TS(configs.enc_in)
        
        # CGA三重注意力融合
        self.cga_fusion = CGAFusion_TS(configs.enc_in)

    def forward(self, x):
        # 先进行小波变换（保持不变）
        (c, d) = self.wavelet(x)
        
        # 生成多尺度特征
        c3 = self.conv3(c)
        c5 = self.conv5(c)
        c7 = self.conv7(c)
        c9 = self.conv9(c)
        
        # MSF多尺度特征融合
        c_msf = self.msf_fusion(c3, c5, c7, c9)
        
        # CGA三重注意力融合原始特征和多尺度特征
        c_enhanced = self.cga_fusion(c, c_msf)
        
        x = c_enhanced

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
    """逆自适应小波块（保持不变）"""
    def __init__(self, configs, input_size):
        super(InverseAdpWaveletBlock, self).__init__()
        self.inverse_wavelet = InverseLiftingScheme(configs.enc_in, input_size=input_size, kernel_size=configs.lifting_kernel_size)

    def forward(self, c, d):
        reconstructed = self.inverse_wavelet(c, d)
        return reconstructed

class ClusteredLinear(nn.Module):
    """聚类线性层"""
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
    YFTT_MSF_DEA: 融合AdaWaveNet、MSFblock和DEANet的时序预测模型
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'super_resolution':
            self.seq_len = self.seq_len // configs.sr_ratio
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # 支持两种参数名称
        n_clusters = getattr(configs, 'n_clusters', getattr(configs, 'n_cluster', 4))
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.series_decomp = series_decomp()
        self.rev_seasonal = RevIN(configs.enc_in)
        self.rev_trend = RevIN(configs.enc_in)
        
        # 使用聚类线性层处理趋势
        self.trend_linear = ClusteredLinear(n_clusters, configs.enc_in, self.seq_len, configs.pred_len)
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len // (2 ** configs.lifting_levels), 
            configs.d_model, 
            configs.embed, 
            configs.freq,
            configs.dropout
        )
        
        # 构建增强的编码器层级（集成MSF和DEA）
        self.encoder_levels = nn.ModuleList()
        self.linear_levels = nn.ModuleList()
        self.coef_linear_levels = nn.ModuleList()
        self.coef_dec_levels = nn.ModuleList()
        input_size = self.seq_len
        
        for i in range(configs.lifting_levels):
            self.encoder_levels.add_module(
                'encoder_level_'+str(i),
                EnhancedAdpWaveletBlock_MSF_DEA(configs, input_size)
            )
            input_size = input_size // 2 
            
            self.linear_levels.add_module(
                'linear_level_'+str(i),
                nn.Sequential(
                    nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=1)
                )
            )
            self.coef_linear_levels.add_module(
                'linear_level_'+str(i),
                nn.Sequential(
                    nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=1)
                )
            )
            self.coef_dec_levels.add_module(
                'linear_level_'+str(i),
                nn.Sequential(
                    nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=1)
                )
            )

        self.input_size = input_size
        
        # 构建解码器层级
        self.decoder_levels = nn.ModuleList()
        for i in range(configs.lifting_levels-1, -1, -1):
            self.decoder_levels.add_module(
                'decoder_level_'+str(i),
                InverseAdpWaveletBlock(configs, input_size=input_size)
            )
            input_size *= 2
        
        self.lowrank_projection = nn.Linear(configs.d_model, self.seq_len // (2 ** configs.lifting_levels), bias=True)

        # 标准编码器
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
        
        # 解码器
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(self.seq_len, configs.pred_len, bias=True)

        self.register_buffer('clusters', None)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, clusters):
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
        
        # 编码阶段（使用MSF和DEA增强的小波块）
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
            
        # 嵌入和编码
        x_enc = x_enc.permute(0,2,1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        
        # 解码阶段
        for dec, x_emb_level, coef_emb_level, c_linear in zip(self.decoder_levels, x_embedding_levels[::-1], coef_embedding_levels[::-1], self.coef_dec_levels[::-1]):
            details = encoded_coefficients.pop()
            details = coef_emb_level + c_linear(details)
            x_dec = x_dec + x_emb_level
            x_dec = dec(x_dec, details)
            
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        
        # 反归一化
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
            return dec_out[:, -self.pred_len:, :]
        
        return None
