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

class dilated_inception(nn.Module):
    """来自Temporal_conv的膨胀卷积模块"""
    def __init__(self, cin, cout, dilation_factor, seq_len):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.padding = 0
        self.seq_len = seq_len
        self.kernel_set = [2, 3, 6, 7]
        # 将通道平均分为N组. N是卷积层的个数
        cout_per_kernel = int(cout / len(self.kernel_set))
        # k个1D因果膨胀卷积
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout_per_kernel, (1, kern), dilation=(1, dilation_factor)))

        # 计算卷积后的时间维度
        min_output_len = self.seq_len - dilation_factor * (self.kernel_set[-1] - 1)
        
        # 计算拼接后的实际通道数
        actual_cout = cout_per_kernel * len(self.kernel_set)
        
        # 简化线性层，直接使用1x1卷积进行维度调整
        self.out_conv = nn.Conv2d(actual_cout, cin, (1, 1))
        
        # 如果需要调整时间维度，使用插值
        self.need_interpolate = min_output_len != seq_len

    def forward(self, input):
        # input: (B, C, N, T)
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]  # 以时间维度最少的特征为标准

        x = torch.cat(x, dim=1)  # 拼接
        x = self.out_conv(x)  # 通道维度调整
        
        # 如果时间维度不匹配，使用插值调整
        if self.need_interpolate and x.size(-1) != self.seq_len:
            x = F.interpolate(x, size=(x.size(2), self.seq_len), mode='bilinear', align_corners=False)
        
        return x

class temporal_conv(nn.Module):
    """来自Temporal_conv的时间卷积模块，使用门控机制"""
    def __init__(self, cin, cout, dilation_factor, seq_len):
        super(temporal_conv, self).__init__()
        self.filter_convs = dilated_inception(cin=cin, cout=cout, dilation_factor=dilation_factor, seq_len=seq_len)
        self.gated_convs = dilated_inception(cin=cin, cout=cout, dilation_factor=dilation_factor, seq_len=seq_len)

    def forward(self, X):
        # X:(B,C,N,T)
        filter = self.filter_convs(X)  # 执行左边的DIL层
        filter = torch.tanh(filter)  # tanh激活函数
        gate = self.gated_convs(X)  # 执行右边的DIL层
        gate = torch.sigmoid(gate)  # sigmoid门控函数
        out = filter * gate  # 逐元素乘法
        return out

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

class series_decomp(nn.Module):
    """序列分解块"""
    def __init__(self, kernel_size=24, stride=1):
        super(series_decomp, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.moving_avg = moving_avg(kernel_size, stride=stride)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class AdpWaveletBlock_Enhanced(nn.Module):
    """增强的自适应小波块，集成时间卷积"""
    def __init__(self, configs, input_size):
        super(AdpWaveletBlock_Enhanced, self).__init__()
        self.regu_details = configs.regu_details
        self.regu_approx = configs.regu_approx
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingScheme(configs.enc_in, k_size=configs.lifting_kernel_size, input_size=input_size)
        self.norm_x = normalization(configs.enc_in)
        self.norm_d = normalization(configs.enc_in)
        
        # 添加时间卷积增强，但只在输入尺寸足够大时使用
        self.use_temporal_conv = input_size >= 24  # 只有当序列长度>=24时才使用时间卷积
        if self.use_temporal_conv:
            self.temporal_conv = temporal_conv(cin=configs.enc_in, cout=configs.enc_in, 
                                             dilation_factor=1, seq_len=input_size // 2)  # 小波后尺寸减半

    def forward(self, x):
        # 先进行小波变换
        (c, d) = self.wavelet(x)
        
        # 对近似系数应用时间卷积增强（仅在序列足够长时）
        if self.use_temporal_conv:
            # 需要调整维度以适配temporal_conv的输入格式 (B,C,N,T)
            B, C, T = c.shape
            c_reshaped = c.unsqueeze(2)  # (B, C, 1, T)
            c_enhanced = self.temporal_conv(c_reshaped)  # (B, C, 1, T)
            c = c_enhanced.squeeze(2)  # (B, C, T)
        
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
    """逆自适应小波块"""
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
    YEF35: 融合AdaWaveNet和Temporal_conv的时序预测模型
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
        self.series_decomp = series_decomp()
        self.rev_seasonal = RevIN(configs.enc_in)
        self.rev_trend = RevIN(configs.enc_in)
        
        # 使用聚类线性层处理趋势
        self.trend_linear = ClusteredLinear(n_clusters, configs.enc_in, self.seq_len, configs.pred_len)
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len // (2 ** configs.lifting_levels), configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # 构建增强的编码器层级（集成时间卷积）
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
                AdpWaveletBlock_Enhanced(configs, input_size)  # 使用增强版本
            )
            in_planes *= 1
            # 小波变换后，时间维度减半
            next_input_size = input_size // 2 
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
            input_size = next_input_size

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
        
        # 编码阶段（使用增强的小波块）
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            # 使用1D卷积处理，输入格式为(B, C, T)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
            
        # Embedding
        x_enc = x_enc.permute(0,2,1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        
        # 解码阶段
        for dec, x_emb_level, coef_emb_level, c_linear in zip(self.decoder_levels, x_embedding_levels[::-1], coef_embedding_levels[::-1], self.coef_dec_levels[::-1]):
            details = encoded_coefficients.pop()
            # 使用1D卷积处理，输入格式为(B, C, T)
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
