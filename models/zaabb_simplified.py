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

class MultiScaleDWConv1D(nn.Module):
    """
    Multi-scale Depthwise Convolution adapted for 1D time series
    """
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            
            if scale[i] == 1:
                conv = nn.Identity()
            else:
                conv = nn.Conv1d(channels, channels,
                                kernel_size=scale[i],
                                padding=scale[i]//2,
                                groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)
            
    def forward(self, x):
        # x: (B, C, L)
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x

class DynamicConv1d(nn.Module):
    """
    Simplified Dynamic Convolution for 1D time series
    """
    def __init__(self, dim, kernel_size=3, reduction_ratio=4, num_groups=2):
        super().__init__()
        self.num_groups = num_groups
        self.K = kernel_size
        
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool1d(output_size=kernel_size)
        
        self.proj = nn.Sequential(
            nn.Conv1d(dim, dim//reduction_ratio, kernel_size=1),
            nn.BatchNorm1d(dim//reduction_ratio),
            nn.GELU(),
            nn.Conv1d(dim//reduction_ratio, dim*num_groups, kernel_size=1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        
        # Generate dynamic weights
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K)

        # Apply dynamic convolution
        x = F.conv1d(x.reshape(1, -1, L),
                     weight=weight,
                     padding=self.K//2,
                     groups=B*C)
        
        return x.reshape(B, C, L)

class Attention1D(nn.Module):
    """
    Simplified Attention for 1D time series
    """
    def __init__(self, dim, num_heads=1, sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Conv1d(dim, dim, kernel_size=1)
        self.kv = nn.Conv1d(dim, dim*2, kernel_size=1)
        
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=sr_ratio+2, stride=sr_ratio, 
                         padding=(sr_ratio+2)//2, groups=dim, bias=False),
                nn.BatchNorm1d(dim),
                nn.GELU(),
            )
        else:
            self.sr = nn.Identity()
            
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        
        q = self.q(x).reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C//self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)
        
        attn = (q @ k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        x = (attn @ v).transpose(-1, -2)
        
        return x.reshape(B, C, L)

class HybridTokenMixer1D(nn.Module):
    """
    Simplified Hybrid Token Mixer for 1D time series
    """
    def __init__(self, dim, kernel_size=3, num_groups=2, num_heads=1, sr_ratio=1):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        # Local unit: Dynamic Convolution
        self.local_unit = DynamicConv1d(
            dim=dim//2, kernel_size=kernel_size, num_groups=num_groups)
        
        # Global unit: Attention
        self.global_unit = Attention1D(
            dim=dim//2, num_heads=num_heads, sr_ratio=sr_ratio)
        
        # Simplified projection
        self.proj = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        # x: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)
        
        # Split into local and global paths
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.local_unit(x1)
        x2 = self.global_unit(x2)
        
        # Combine and project
        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x) + x  # Skip connection
        
        # Convert back to (B, L, C)
        return x.permute(0, 2, 1)

class SimplifiedFeedForward(nn.Module):
    """
    Simplified Multi-scale Feed Forward Network
    """
    def __init__(self, in_features, hidden_features=None, drop=0):
        super().__init__()
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dwconv = MultiScaleDWConv1D(hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: (B, L, C)
        x = self.fc1(x)
        x = self.act(x)
        
        # Multi-scale processing
        x_conv = x.permute(0, 2, 1)  # (B, C, L)
        x_conv = self.dwconv(x_conv) + x_conv
        x = x_conv.permute(0, 2, 1)  # (B, L, C)
        
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class SimplifiedTransXNetEncoder(nn.Module):
    """
    Simplified TransXNet-inspired encoder
    """
    def __init__(self, configs):
        super(SimplifiedTransXNetEncoder, self).__init__()
        
        # Hybrid Token Mixer
        self.token_mixer = HybridTokenMixer1D(
            dim=configs.d_model,
            kernel_size=3,
            num_groups=2,
            num_heads=configs.n_heads,
            sr_ratio=2
        )
        
        # Simplified Feed Forward
        self.mlp = SimplifiedFeedForward(
            in_features=configs.d_model,
            hidden_features=configs.d_ff,
            drop=configs.dropout
        )
        
        # Standard Transformer layers
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
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        
        # Simplified fusion
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, attn_mask=None):
        # Two parallel paths with simplified fusion
        # Path 1: Hybrid Token Mixer + MLP
        mixer_out = self.token_mixer(x)
        mixer_out = self.norm1(mixer_out + x)
        mixer_out = self.mlp(mixer_out) + mixer_out
        
        # Path 2: Standard Transformer
        transformer_out, attns = self.transformer_encoder(x, attn_mask=attn_mask)
        transformer_out = self.norm2(transformer_out)
        
        # Simple weighted fusion
        weight = torch.sigmoid(self.fusion_weight)
        output = weight * mixer_out + (1 - weight) * transformer_out
        
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
    ZAABB_Simplified: Simplified AdaWaveNet with essential TransXNet components
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

        # Simplified TransXNet-inspired encoder
        self.encoder = SimplifiedTransXNetEncoder(configs)
        
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
        # Encoding
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x_enc, r, details = l(x_enc)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x_enc))
        
        # Embedding
        x_enc = x_enc.permute(0,2,1)
        enc_out = self.enc_embedding(x_enc, None)
        
        # Simplified TransXNet encoder processing
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






































