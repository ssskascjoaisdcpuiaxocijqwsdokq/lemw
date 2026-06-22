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

class EfficientMultiScaleConv(nn.Module):
    """高效多尺度卷积模块 - 性能优化版本"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 使用分组卷积减少计算量
        self.conv3 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim//4, bias=False)
        self.conv5 = nn.Conv1d(dim, dim, 5, padding=2, groups=dim//4, bias=False)
        self.conv7 = nn.Conv1d(dim, dim, 7, padding=3, groups=dim//4, bias=False)
        
        # 轻量级融合
        self.fusion = nn.Conv1d(dim * 3, dim, 1, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        # x: (B, C, T)
        identity = x
        
        # 多尺度特征提取
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        
        # 特征融合
        out = torch.cat([x3, x5, x7], dim=1)
        out = self.fusion(out)
        out = self.act(out)
        
        # LayerNorm
        out = out.transpose(1, 2)  # (B, T, C)
        out = self.norm(out)
        out = out.transpose(1, 2)  # (B, C, T)
        
        # 残差连接
        return out + identity

class LightweightAttention(nn.Module):
    """轻量级注意力机制 - 性能优化版本"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 使用单个线性层生成QKV，减少参数
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # 位置编码 - 简化版本
        self.pos_emb = nn.Parameter(torch.randn(1, 1000, dim) * 0.02)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # 添加位置编码
        if T <= self.pos_emb.size(1):
            x = x + self.pos_emb[:, :T, :]
        
        # 生成QKV
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 输出投影
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class OptimizedFFN(nn.Module):
    """优化的前馈网络"""
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2  # 减少隐藏层维度
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

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

class OptimizedAdpWaveletBlock(nn.Module):
    """优化的自适应小波块"""
    def __init__(self, configs, input_size):
        super(OptimizedAdpWaveletBlock, self).__init__()
        self.regu_details = configs.regu_details
        self.regu_approx = configs.regu_approx
        
        # 保持原有的小波结构不变
        self.wavelet = LiftingScheme(configs.enc_in, k_size=configs.lifting_kernel_size, input_size=input_size)
        self.norm_x = normalization(configs.enc_in)
        self.norm_d = normalization(configs.enc_in)
        
        # 添加高效多尺度卷积
        self.multi_conv = EfficientMultiScaleConv(configs.enc_in)

    def forward(self, x):
        # 小波变换（保持不变）
        (c, d) = self.wavelet(x)
        
        # 多尺度卷积增强
        c_enhanced = self.multi_conv(c)
        
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
    YFTT_Optimized: 性能优化版本的时序预测模型
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
        
        # 轻量级嵌入
        embed_dim = configs.d_model // 2  # 减少嵌入维度
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len // (2 ** configs.lifting_levels), 
            embed_dim, 
            configs.embed, 
            configs.freq,
            configs.dropout
        )
        
        # 构建优化的编码器层级
        self.encoder_levels = nn.ModuleList()
        self.linear_levels = nn.ModuleList()
        self.coef_linear_levels = nn.ModuleList()
        self.coef_dec_levels = nn.ModuleList()
        input_size = self.seq_len
        
        for i in range(configs.lifting_levels):
            self.encoder_levels.add_module(
                'encoder_level_'+str(i),
                OptimizedAdpWaveletBlock(configs, input_size)
            )
            input_size = input_size // 2 
            
            # 使用更轻量的线性层
            self.linear_levels.add_module(
                'linear_level_'+str(i),
                nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=1, bias=False)
            )
            self.coef_linear_levels.add_module(
                'linear_level_'+str(i),
                nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=1, bias=False)
            )
            self.coef_dec_levels.add_module(
                'linear_level_'+str(i),
                nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=1, bias=False)
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
        
        # 轻量级投影层
        self.lowrank_projection = nn.Linear(embed_dim, self.seq_len // (2 ** configs.lifting_levels), bias=True)

        # 优化的注意力机制
        self.lightweight_attn = LightweightAttention(
            dim=embed_dim,
            num_heads=configs.n_heads // 2,  # 减少注意力头数
            dropout=configs.dropout
        )
        
        # 优化的FFN
        self.ffn = OptimizedFFN(embed_dim, embed_dim * 2, configs.dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 简化的编码器
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), embed_dim, configs.n_heads // 2),
                    embed_dim,
                    embed_dim * 2,  # 减少FFN维度
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(max(1, configs.e_layers // 2))  # 减少编码器层数
            ],
            norm_layer=torch.nn.LayerNorm(embed_dim)
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
        
        # 编码阶段
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
            
        # 嵌入和注意力
        x_enc = x_enc.permute(0,2,1)
        enc_out = self.enc_embedding(x_enc, None)
        
        # 轻量级注意力增强
        enc_out_enhanced = self.lightweight_attn(enc_out)
        enc_out = self.norm1(enc_out + enc_out_enhanced)
        
        # FFN
        ffn_out = self.ffn(enc_out)
        enc_out = self.norm2(enc_out + ffn_out)
        
        # 传统编码器（简化版）
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








































