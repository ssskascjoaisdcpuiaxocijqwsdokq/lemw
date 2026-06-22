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

class AdvancedFrequencyAttention(nn.Module):
    """
    高级频域注意力机制，多层次频域处理
    """
    def __init__(self, d_model, freq_num=48, num_heads=8):
        super(AdvancedFrequencyAttention, self).__init__()
        self.d_model = d_model
        self.freq_num = freq_num
        self.num_heads = num_heads
        
        # 多层频域处理
        self.freq_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(3)
        ])
        
        # 不同尺度的频域卷积
        self.freq_convs = nn.ModuleList([
            nn.Conv1d(d_model // 2, d_model // 2, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]
        ])
        
        # 频域特征融合
        self.freq_fusion = nn.Sequential(
            nn.Linear(d_model // 2 * 4, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 自适应频域权重
        self.freq_weights = nn.Parameter(torch.ones(4) / 4)
        
        # 增强的多头注意力
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        
        # 交叉注意力（频域-时域）
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 增强的前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        residual = x
        
        # 频域变换
        x_freq = torch.fft.rfft(x, dim=1)
        x_freq_real = x_freq.real
        
        # 保留更多频率成分
        if x_freq_real.size(1) > self.freq_num:
            x_freq_real = x_freq_real[:, :self.freq_num, :]
        
        # 多层频域处理
        freq_features = []
        for i, (layer, conv) in enumerate(zip(self.freq_layers, self.freq_convs)):
            freq_proj = layer(x_freq_real)  # (B, freq_num, C//2)
            freq_proj = freq_proj.permute(0, 2, 1)  # (B, C//2, freq_num)
            freq_conv = F.gelu(conv(freq_proj))
            freq_conv = freq_conv.permute(0, 2, 1)  # (B, freq_num, C//2)
            freq_features.append(freq_conv)
        
        # 加权融合频域特征
        weights = F.softmax(self.freq_weights, dim=0)
        freq_fused = sum(w * feat for w, feat in zip(weights, freq_features))
        
        # 确保维度正确
        freq_concat = torch.cat(freq_features, dim=-1)
        freq_enhanced = self.freq_fusion(freq_concat)
        
        # 逆变换回时域
        if freq_enhanced.size(1) < L // 2 + 1:
            pad_size = L // 2 + 1 - freq_enhanced.size(1)
            freq_enhanced = F.pad(freq_enhanced, (0, 0, 0, pad_size))
        
        freq_complex = torch.complex(freq_enhanced, torch.zeros_like(freq_enhanced))
        enhanced_x = torch.fft.irfft(freq_complex, n=L, dim=1)
        
        # 第一层：频域增强 + 残差
        x = self.norm1(enhanced_x + residual)
        
        # 第二层：自注意力
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm2(x + attn_out)
        
        # 第三层：交叉注意力（频域特征作为key和value）
        cross_out, _ = self.cross_attention(x, enhanced_x, enhanced_x)
        x = self.norm3(x + cross_out)
        
        # 前馈网络
        ffn_out = self.ffn(x)
        x = x + ffn_out
        
        return x

class MultiScaleConvolution(nn.Module):
    """
    多尺度卷积模块
    """
    def __init__(self, d_model):
        super(MultiScaleConvolution, self).__init__()
        self.d_model = d_model
        
        # 不同尺度的卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model // 4, kernel_size=k, padding=k//2, groups=d_model//16)
            for k in [3, 5, 7, 9]
        ])
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, L, C) -> (B, C, L)
        x_conv = x.permute(0, 2, 1)
        
        # 多尺度卷积
        conv_outs = []
        for conv in self.convs:
            conv_out = F.gelu(conv(x_conv))
            conv_outs.append(conv_out)
        
        # 拼接并转回 (B, L, C)
        fused = torch.cat(conv_outs, dim=1).permute(0, 2, 1)
        
        # 特征融合和残差连接
        output = self.fusion(fused)
        output = self.norm(output + x)
        
        return output

class UltimateEncoder(nn.Module):
    """
    终极编码器，集成所有优化技术
    """
    def __init__(self, configs):
        super(UltimateEncoder, self).__init__()
        
        # 高级频域注意力
        self.freq_attention = AdvancedFrequencyAttention(configs.d_model, num_heads=configs.n_heads)
        
        # 多尺度卷积
        self.multi_scale_conv = MultiScaleConvolution(configs.d_model)
        
        # 标准Transformer编码器（增加层数）
        self.transformer_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers + 1)  # 增加一层
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # 三路特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(configs.d_model * 3, configs.d_model * 2),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.LayerNorm(configs.d_model)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(configs.d_model * 3, configs.d_model),
            nn.Sigmoid()
        )
        
        self.final_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x, attn_mask=None):
        # 保存原始输入
        residual = x
        
        # 三个并行分支
        freq_out = self.freq_attention(x)
        conv_out = self.multi_scale_conv(x)
        transformer_out, attns = self.transformer_encoder(x, attn_mask=attn_mask)
        
        # 三路特征拼接
        combined = torch.cat([freq_out, conv_out, transformer_out], dim=-1)
        
        # 门控融合
        gate_weights = self.gate(combined)
        fused = self.feature_fusion(combined)
        
        # 应用门控和残差连接
        output = gate_weights * fused + (1 - gate_weights) * residual
        output = self.final_norm(output)
        output = self.dropout(output)
        
        return output, attns

class EnhancedPredictionHead(nn.Module):
    """
    增强的预测头
    """
    def __init__(self, seq_len, pred_len, dropout=0.1):
        super(EnhancedPredictionHead, self).__init__()
        
        # 多层预测网络
        self.predictor = nn.Sequential(
            nn.Linear(seq_len, seq_len // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(seq_len // 2, seq_len // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(seq_len // 4, pred_len)
        )
        
        # 残差连接（如果维度匹配）
        self.residual_proj = nn.Linear(seq_len, pred_len) if seq_len != pred_len else nn.Identity()
        
    def forward(self, x):
        pred_out = self.predictor(x)
        
        # 残差连接
        if hasattr(self.residual_proj, 'weight'):
            residual = self.residual_proj(x)
            pred_out = pred_out + residual * 0.1  # 小权重残差
        
        return pred_out

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

class series_decomp(nn.Module):
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
    ZAAA_Ultimate: 终极优化版本，目标实现MSE和MAE < 0.45
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
        
        # Wavelet levels
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
        
        # Decoder levels
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

        # 终极编码器
        self.encoder = UltimateEncoder(configs)
        
        # 增强预测头
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = EnhancedPredictionHead(self.seq_len, configs.pred_len, configs.dropout)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(self.seq_len, self.seq_len, bias=True)
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(self.seq_len, self.seq_len, bias=True)
        elif self.task_name == 'super_resolution':
            self.projection = nn.Linear(configs.pred_len, configs.pred_len, bias=True)

        self.register_buffer('clusters', None)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, clusters):
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        
        # Normalization
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
        
        # 终极编码器处理
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        
        # Decoding
        for dec, x_emb_level, coef_emb_level, c_linear in zip(self.decoder_levels, x_embedding_levels[::-1], coef_embedding_levels[::-1], self.coef_dec_levels[::-1]):
            details = encoded_coefficients.pop()
            details = coef_emb_level + c_linear(details)
            x_dec = x_dec + x_emb_level
            x_dec = dec(x_dec, details)
            
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        
        # De-Normalization
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
