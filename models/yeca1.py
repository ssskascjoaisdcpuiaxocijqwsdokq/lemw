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
    def __init__(self, channel=512, kernel_size=3, dropout=0.1, use_max=True, adaptive_kernel=False):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channel)
        self.eca_attention = ECAAttention1D(channel=channel, kernel_size=kernel_size, dropout=dropout, use_max=use_max, adaptive_kernel=adaptive_kernel)
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

    def forward(self, x):
        residual = x
        x_norm = self.pre_norm(x.transpose(1, 2)).transpose(1, 2)
        attention_out = self.eca_attention(x_norm)
        enhanced_out = self.feature_enhance(attention_out)
        output = residual * self.residual_scale + enhanced_out * self.attention_scale
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
    YECA1: 优化版 AdaWaveNet + ECA 注意力
    保持小波结构不变，在分解后引入增强ECA块
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

        self.trend_linear = ClusteredLinear(configs.n_clusters, configs.enc_in, self.seq_len, configs.pred_len)

        # ECA块（可配置）
        eca_kernel_size = getattr(configs, 'eca_kernel_size', 3)
        eca_dropout = getattr(configs, 'eca_dropout', configs.dropout)
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

        self.register_buffer('clusters', None)

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

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, clusters):
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
        moving_mean_out = self.trend_linear(moving_mean.permute(0,2,1), self.clusters).permute(0,2,1)
        moving_mean_out = self.rev_trend(moving_mean_out, 'denorm')
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
        moving_mean_out = self.trend_linear(moving_mean.permute(0,2,1), self.clusters).permute(0,2,1)
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
        moving_mean_out = self.trend_linear(moving_mean.permute(0,2,1), self.clusters).permute(0,2,1)
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



