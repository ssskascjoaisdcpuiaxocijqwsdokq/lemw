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

class EnhancedDCTAttention(nn.Module):
    """
    增强的DCT注意力机制，包含多尺度频域处理和自适应权重
    """
    def __init__(self, d_model, freq_num=32, num_heads=8):
        super(EnhancedDCTAttention, self).__init__()
        self.d_model = d_model
        self.freq_num = freq_num
        self.num_heads = num_heads
        
        # 多尺度频域处理
        self.freq_proj1 = nn.Linear(d_model, d_model // 2)
        self.freq_proj2 = nn.Linear(d_model, d_model // 4)
        
        # 不同尺度的卷积
        self.freq_conv1 = nn.Conv1d(d_model // 2, d_model // 2, 3, padding=1)
        self.freq_conv2 = nn.Conv1d(d_model // 4, d_model // 4, 5, padding=2)
        self.freq_conv3 = nn.Conv1d(d_model // 4, d_model // 4, 7, padding=3)
        
        # 特征融合
        self.fusion = nn.Linear(d_model // 2 + d_model // 4 + d_model // 4, d_model)
        
        # 自适应权重生成
        self.weight_gen = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 3),
            nn.Softmax(dim=-1)
        )
        
        # 增强的多头注意力
        self.attention = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True, dropout=0.1)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        residual = x
        
        # 频域增强
        x_freq = torch.fft.rfft(x, dim=1)  # (B, L//2+1, C)
        x_freq_real = x_freq.real
        
        # 保留不同数量的频率成分
        if x_freq_real.size(1) > self.freq_num:
            x_freq_real = x_freq_real[:, :self.freq_num, :]
        
        # 多尺度频域处理
        freq1 = self.freq_proj1(x_freq_real)  # (B, freq_num, C//2)
        freq2 = self.freq_proj2(x_freq_real)  # (B, freq_num, C//4)
        
        # 不同尺度的卷积处理
        freq1 = freq1.permute(0, 2, 1)  # (B, C//2, freq_num)
        freq2 = freq2.permute(0, 2, 1)  # (B, C//4, freq_num)
        
        freq1_conv = F.gelu(self.freq_conv1(freq1))
        freq2_conv1 = F.gelu(self.freq_conv2(freq2))
        freq2_conv2 = F.gelu(self.freq_conv3(freq2))
        
        # 转回原始维度
        freq1_conv = freq1_conv.permute(0, 2, 1)  # (B, freq_num, C//2)
        freq2_conv1 = freq2_conv1.permute(0, 2, 1)  # (B, freq_num, C//4)
        freq2_conv2 = freq2_conv2.permute(0, 2, 1)  # (B, freq_num, C//4)
        
        # 特征融合
        freq_fused = torch.cat([freq1_conv, freq2_conv1, freq2_conv2], dim=-1)
        freq_enhanced = self.fusion(freq_fused)  # (B, freq_num, C)
        
        # 逆变换回时域
        if freq_enhanced.size(1) < L // 2 + 1:
            pad_size = L // 2 + 1 - freq_enhanced.size(1)
            freq_enhanced = F.pad(freq_enhanced, (0, 0, 0, pad_size))
        
        freq_complex = torch.complex(freq_enhanced, torch.zeros_like(freq_enhanced))
        enhanced_x = torch.fft.irfft(freq_complex, n=L, dim=1)  # (B, L, C)
        
        # 自适应权重
        weights = self.weight_gen(x.mean(dim=1))  # (B, 3)
        enhanced_x = weights[:, 0:1, None] * x + weights[:, 1:2, None] * enhanced_x + weights[:, 2:3, None] * residual
        
        # 第一个残差连接和归一化
        x = self.norm1(enhanced_x + residual)
        
        # 多头注意力
        attn_out, _ = self.attention(x, x, x)
        x = self.norm2(x + attn_out)
        
        # 前馈网络
        ffn_out = self.ffn(x)
        x = x + ffn_out
        
        return x

class MultiScaleFeatureFusion(nn.Module):
    """
    多尺度特征融合模块
    """
    def __init__(self, d_model):
        super(MultiScaleFeatureFusion, self).__init__()
        self.d_model = d_model
        
        # 不同尺度的卷积
        self.conv1 = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model//8)
        self.conv2 = nn.Conv1d(d_model, d_model, 5, padding=2, groups=d_model//8)
        self.conv3 = nn.Conv1d(d_model, d_model, 7, padding=3, groups=d_model//8)
        
        # 特征融合权重
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, L, C) -> (B, C, L)
        x_conv = x.permute(0, 2, 1)
        
        # 多尺度卷积
        conv1_out = F.gelu(self.conv1(x_conv))
        conv2_out = F.gelu(self.conv2(x_conv))
        conv3_out = F.gelu(self.conv3(x_conv))
        
        # 加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = weights[0] * conv1_out + weights[1] * conv2_out + weights[2] * conv3_out
        
        # 转回 (B, L, C)
        fused = fused.permute(0, 2, 1)
        
        # 输出投影和残差连接
        output = self.output_proj(fused)
        output = self.norm(output + x)
        
        return output

class EnhancedEncoder(nn.Module):
    """
    增强编码器，结合频域处理、多尺度融合和注意力机制
    """
    def __init__(self, configs):
        super(EnhancedEncoder, self).__init__()
        
        # 增强的DCT频域注意力
        self.dct_attention = EnhancedDCTAttention(configs.d_model, num_heads=configs.n_heads)
        
        # 多尺度特征融合
        self.multi_scale_fusion = MultiScaleFeatureFusion(configs.d_model)
        
        # 标准Transformer编码器
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
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # 特征融合和输出
        self.feature_fusion = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.LayerNorm(configs.d_model)
        )
        
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x, attn_mask=None):
        # 保存原始输入
        residual = x
        
        # DCT频域增强
        dct_out = self.dct_attention(x)
        
        # 多尺度特征融合
        multi_scale_out = self.multi_scale_fusion(dct_out)
        
        # Transformer处理
        transformer_out, attns = self.transformer_encoder(multi_scale_out, attn_mask=attn_mask)
        
        # 特征融合：结合频域增强和Transformer输出
        fused_features = torch.cat([dct_out, transformer_out], dim=-1)
        output = self.feature_fusion(fused_features)
        
        # 最终残差连接
        output = output + residual
        output = self.dropout(output)
        
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

class AdaptiveLoss(nn.Module):
    """
    自适应损失函数，结合MSE和MAE，并根据预测误差动态调整权重
    """
    def __init__(self, alpha=0.5, beta=1.0):
        super(AdaptiveLoss, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))  # MSE权重
        self.beta = nn.Parameter(torch.tensor(beta))    # MAE权重
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        mae = self.mae_loss(pred, target)
        
        # 自适应权重：当MSE较大时增加MSE权重，当MAE较大时增加MAE权重
        mse_weight = torch.sigmoid(self.alpha)
        mae_weight = torch.sigmoid(self.beta)
        
        # 归一化权重
        total_weight = mse_weight + mae_weight
        mse_weight = mse_weight / total_weight
        mae_weight = mae_weight / total_weight
        
        adaptive_loss = mse_weight * mse + mae_weight * mae
        
        return adaptive_loss, mse, mae

class Model(nn.Module):
    """
    ZAAA_Enhanced: 增强版本，包含多尺度特征融合、增强频域注意力和自适应损失
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

        # 增强编码器（替代复杂的脉冲神经网络）
        self.encoder = EnhancedEncoder(configs)
        
        # 增强预测头
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Sequential(
                nn.Linear(self.seq_len, self.seq_len // 2),
                nn.GELU(),
                nn.Dropout(configs.dropout * 0.5),
                nn.Linear(self.seq_len // 2, configs.pred_len),
                nn.Dropout(configs.dropout * 0.3)
            )
        elif self.task_name == 'imputation':
            self.projection = nn.Sequential(
                nn.Linear(self.seq_len, self.seq_len),
                nn.GELU(),
                nn.Dropout(configs.dropout * 0.5),
                nn.Linear(self.seq_len, self.seq_len)
            )
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Sequential(
                nn.Linear(self.seq_len, self.seq_len),
                nn.GELU(),
                nn.Dropout(configs.dropout * 0.5),
                nn.Linear(self.seq_len, self.seq_len)
            )
        elif self.task_name == 'super_resolution':
            self.projection = nn.Sequential(
                nn.Linear(configs.pred_len, configs.pred_len * 2),
                nn.GELU(),
                nn.Dropout(configs.dropout * 0.5),
                nn.Linear(configs.pred_len * 2, configs.pred_len)
            )
        
        # 自适应损失函数
        self.adaptive_loss = AdaptiveLoss()

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
        
        # 轻量级编码器处理
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
        
        return None
