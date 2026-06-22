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
# 简化的DropPath实现，避免timm依赖
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)

class MultiDWConv_TS(nn.Module):
    """多尺度深度卷积模块 - 适配时序数据，简化版本"""
    def __init__(self, dim=768):
        super().__init__()
        self.dim = dim

        # 使用深度可分离卷积，避免通道数问题
        self.dwconv1 = nn.Conv1d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv2 = nn.Conv1d(dim, dim, 5, 1, 2, bias=True, groups=dim)
        self.dwconv3 = nn.Conv1d(dim, dim, 7, 1, 3, bias=True, groups=dim)
        
        self.pointwise = nn.Conv1d(dim * 3, dim, 1)  # 点卷积融合
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim)  # 使用LayerNorm替代BatchNorm

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        
        # 多尺度深度卷积
        x1 = self.dwconv1(x)  # (B, C, T)
        x2 = self.dwconv2(x)  # (B, C, T)
        x3 = self.dwconv3(x)  # (B, C, T)
        
        # 拼接多尺度特征
        out = torch.cat([x1, x2, x3], dim=1)  # (B, 3C, T)
        out = self.pointwise(out)  # (B, C, T)
        out = self.act(out)
        
        # LayerNorm需要转置
        out = out.transpose(1, 2)  # (B, T, C)
        out = self.norm(out)
        out = out.transpose(1, 2)  # (B, C, T)
        
        # 残差连接
        out = out + x
        return out

class MRFP_TS(nn.Module):
    """多尺度感受野处理模块 - 适配时序数据"""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = MultiDWConv_TS(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        x = self.fc1(x)  # (B, T, C) -> (B, T, hidden_features)
        
        # 转换为卷积格式
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = self.dwconv(x)  # (B, C, T) -> (B, C, T)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  # (B, T, hidden_features) -> (B, T, out_features)
        x = self.drop(x)
        return x

class RoPE_TS(torch.nn.Module):
    """旋转位置编码 - 适配时序数据"""
    def __init__(self, seq_len, feature_dim, base=10000):
        super(RoPE_TS, self).__init__()
        
        k_max = feature_dim // 2
        assert feature_dim % 2 == 0
        
        # 为1D时序数据生成角度
        theta_ks = 1 / (base ** (torch.arange(k_max, dtype=torch.float32) / k_max))
        positions = torch.arange(seq_len, dtype=torch.float32)
        angles = positions.unsqueeze(-1) * theta_ks.unsqueeze(0)  # (seq_len, k_max)
        
        # 旋转矩阵
        rotations_re = torch.cos(angles).unsqueeze(-1)  # (seq_len, k_max, 1)
        rotations_im = torch.sin(angles).unsqueeze(-1)  # (seq_len, k_max, 1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)  # (seq_len, k_max, 2)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        # x: (B, T, C)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        
        B, T, C = x.shape
        x = x.reshape(B, T, -1, 2)  # (B, T, C/2, 2)
        x = torch.view_as_complex(x)  # (B, T, C/2)
        
        # 应用旋转
        rotations = torch.view_as_complex(self.rotations[:T])  # (T, C/2)
        pe_x = rotations.unsqueeze(0) * x  # (1, T, C/2) * (B, T, C/2) = (B, T, C/2)
        
        return torch.view_as_real(pe_x).flatten(-2)  # (B, T, C)

class LinearAttention_TS(nn.Module):
    """线性注意力 - 适配时序数据"""
    def __init__(self, dim, seq_len, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)  # 1D卷积
        self.rope = RoPE_TS(seq_len, dim)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        
        qk = self.qk(x).reshape(B, T, 2, C).permute(2, 0, 1, 3)  # (2, B, T, C)
        q, k, v = qk[0], qk[1], x  # (B, T, C)
        
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        
        # 应用RoPE
        q_rope = self.rope(q).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, h, T, d)
        k_rope = self.rope(k).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, h, T, d)
        q = q.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, h, T, d)
        k = k.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, h, T, d)
        v = v.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, h, T, d)
        
        # 线性注意力计算
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)  # (B, h, T, 1)
        kv = (k_rope.transpose(-2, -1) * (T ** -0.5)) @ (v * (T ** -0.5))  # (B, h, d, d)
        x = q_rope @ kv * z  # (B, h, T, d)
        
        x = x.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        
        # 添加位置编码
        v_1d = v.transpose(1, 2).reshape(B, T, C).transpose(1, 2)  # (B, C, T)
        x = x + self.lepe(v_1d).transpose(1, 2)  # (B, T, C)
        
        return x

class MLLABlock_TS(nn.Module):
    """MLLA块 - 适配时序数据"""
    def __init__(self, dim, seq_len, num_heads, mlp_ratio=4., qkv_bias=True, 
                 drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention_TS(dim=dim, seq_len=seq_len, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = MRFP_TS(in_features=dim, hidden_features=int(dim * mlp_ratio), 
                          act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        
        # 第一个卷积位置编码
        x = x + self.cpe1(x.transpose(1, 2)).transpose(1, 2)
        shortcut = x
        
        # 归一化
        x = self.norm1(x)
        
        # MLLA块
        act_res = self.act(self.act_proj(x))  # 右分支
        
        # 左分支
        x = self.in_proj(x)  # (B, T, C)
        x = self.act(self.dwc(x.transpose(1, 2))).transpose(1, 2)  # 1D卷积
        x = self.attn(x)  # 线性注意力
        x = self.out_proj(x * act_res)  # 门控融合
        
        # 残差连接
        x = shortcut + self.drop_path(x)
        
        # 第二个卷积位置编码
        x = x + self.cpe2(x.transpose(1, 2)).transpose(1, 2)
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class moving_avg(nn.Module):
    """移动平均块，用于突出时间序列的趋势"""
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # x - B, C, L
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2))
        end = x[:, :, -1:].repeat(1, 1, math.floor((self.kernel_size - 1) // 2))
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x

class moving_avg_imputation(nn.Module):
    """移动平均块修改版，忽略移动窗口中的零值"""
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
    """序列分解块"""
    def __init__(self, kernel_size=24, stride=1, imputation=False):
        super(series_decomp, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.moving_avg = moving_avg(kernel_size, stride=stride) if not imputation else moving_avg_imputation(self.kernel_size, self.stride)

    def forward(self, x):
        moving_mean = self.moving_avg(x) 
        res = x - moving_mean
        return res, moving_mean

class EnhancedAdpWaveletBlock_V2(nn.Module):
    """增强的自适应小波块V2 - 集成VITComer多尺度卷积"""
    def __init__(self, configs, input_size):
        super(EnhancedAdpWaveletBlock_V2, self).__init__()
        self.regu_details = configs.regu_details
        self.regu_approx = configs.regu_approx
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        # 保持原有的小波结构不变
        self.wavelet = LiftingScheme(configs.enc_in, k_size=configs.lifting_kernel_size, input_size=input_size)
        self.norm_x = normalization(configs.enc_in)
        self.norm_d = normalization(configs.enc_in)
        
        # 添加多尺度深度卷积增强
        self.multi_dwconv = MultiDWConv_TS(configs.enc_in)

    def forward(self, x):
        # 先进行小波变换（保持不变）
        (c, d) = self.wavelet(x)
        
        # 对近似系数应用多尺度深度卷积增强
        c_enhanced = self.multi_dwconv(c)
        
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
    YFTT_V2: 融合AdaWaveNet、VITComer多尺度卷积和MLLAttention的时序预测模型
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'super_resolution':
            self.seq_len = self.seq_len // configs.sr_ratio
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # 支持两种参数名称：n_clusters 和 n_cluster
        n_clusters = getattr(configs, 'n_clusters', getattr(configs, 'n_cluster', 4))
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.series_decomp = series_decomp(imputation = self.task_name=='imputation')
        self.rev_seasonal = RevIN(configs.enc_in)
        self.rev_trend = RevIN(configs.enc_in)
        
        # 使用聚类线性层处理趋势
        self.trend_linear = ClusteredLinear(n_clusters, configs.enc_in, self.seq_len, configs.pred_len)
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len // (2 ** configs.lifting_levels), configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # 构建增强的编码器层级（集成VITComer多尺度卷积）
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
                EnhancedAdpWaveletBlock_V2(configs, input_size)  # 使用V2增强版本
            )
            in_planes *= 1
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
            in_planes //= 1
            input_size *= 2
        
        if self.task_name == "super_resolution":
            self.lowrank_projection = nn.Linear(configs.d_model, self.pred_len // (2 ** configs.lifting_levels), bias=True)
        else:
            self.lowrank_projection = nn.Linear(configs.d_model, self.seq_len // (2 ** configs.lifting_levels), bias=True)

        # 增强的Encoder，集成MLLAttention
        self.mlla_block = MLLABlock_TS(
            dim=configs.d_model,
            seq_len=self.seq_len // (2 ** configs.lifting_levels),
            num_heads=configs.n_heads,
            mlp_ratio=4.0,
            drop=configs.dropout,
            drop_path=0.1
        )
        
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
        
        # 编码阶段（使用增强的小波块V2）
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
            
        # Embedding
        x_enc = x_enc.permute(0,2,1)
        enc_out = self.enc_embedding(x_enc, None)
        
        # 应用MLLAttention增强
        enc_out_enhanced = self.mlla_block(enc_out)
        enc_out = enc_out + enc_out_enhanced  # 残差连接
        
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        
        # 解码阶段
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
        
        # 其他任务类型可以根据需要添加
        return None
