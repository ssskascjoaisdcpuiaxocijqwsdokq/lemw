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

class SpikeConv1D(nn.Module):
    """
    1D Spike Convolution adapted for time series
    """
    def __init__(self, conv, step=2):
        super(SpikeConv1D, self).__init__()
        self.conv = conv
        self.step = step

    def forward(self, x):
        # x: (B, C, L) or (step, B, C, L)
        if len(x.shape) == 3:
            x = x.unsqueeze(0).repeat(self.step, 1, 1, 1)  # (step, B, C, L)
        
        outputs = []
        for t in range(self.step):
            out = self.conv(x[t])
            outputs.append(out)
        
        return torch.stack(outputs, dim=0)  # (step, B, C_out, L_out)

class SpikePool1D(nn.Module):
    """
    1D Spike Pooling adapted for time series
    """
    def __init__(self, pool, step=2):
        super(SpikePool1D, self).__init__()
        self.pool = pool
        self.step = step

    def forward(self, x):
        # x: (B, C, L) or (step, B, C, L)
        if len(x.shape) == 3:
            x = x.unsqueeze(0).repeat(self.step, 1, 1, 1)  # (step, B, C, L)
        
        outputs = []
        for t in range(self.step):
            out = self.pool(x[t])
            outputs.append(out)
        
        return torch.stack(outputs, dim=0)  # (step, B, C_out, L_out)

class myBatchNorm1d(nn.Module):
    """
    Temporal BatchNorm for spike sequences
    """
    def __init__(self, bn, step=2):
        super(myBatchNorm1d, self).__init__()
        self.bn = bn
        self.step = step

    def forward(self, x):
        # x: (step, B, C, L)
        outputs = []
        for t in range(self.step):
            out = self.bn(x[t])
            outputs.append(out)
        
        return torch.stack(outputs, dim=0)

class DctSpatialLIF1D(nn.Module):
    """
    DCT Spatial LIF adapted for 1D time series
    """
    def __init__(self, step=2, channel=64, length=96, freq_num=32, reduction=16):
        super(DctSpatialLIF1D, self).__init__()
        self.step = step
        self.channel = channel
        self.length = length
        self.freq_num = min(freq_num, length // 2)
        
        # DCT频域处理
        self.dct_conv = nn.Conv1d(channel, channel // reduction, 1, bias=False)
        self.freq_conv = nn.Conv1d(channel // reduction, channel // reduction, 3, padding=1, bias=False)
        self.idct_conv = nn.Conv1d(channel // reduction, channel, 1, bias=False)
        
        # LIF参数
        self.threshold = nn.Parameter(torch.ones(channel) * 0.5)
        self.leak = nn.Parameter(torch.ones(channel) * 0.2)
        
        # 时间记忆 - 改为参数以避免梯度问题
        self.membrane = None
        
    def dct_transform(self, x):
        """简化的DCT变换"""
        # x: (B, C, L)
        B, C, L = x.shape
        
        # 频域变换 (简化版本)
        x_freq = torch.fft.rfft(x, dim=-1)
        x_freq_real = x_freq.real
        
        # 只保留低频成分
        if x_freq_real.size(-1) > self.freq_num:
            x_freq_real = x_freq_real[..., :self.freq_num]
            
        return x_freq_real
    
    def idct_transform(self, x_freq, target_length):
        """简化的逆DCT变换"""
        # 补零到目标长度
        if x_freq.size(-1) < target_length // 2 + 1:
            pad_size = target_length // 2 + 1 - x_freq.size(-1)
            x_freq = F.pad(x_freq, (0, pad_size))
        
        # 转换为复数并逆变换
        x_freq_complex = torch.complex(x_freq, torch.zeros_like(x_freq))
        x_reconstructed = torch.fft.irfft(x_freq_complex, n=target_length, dim=-1)
        
        return x_reconstructed

    def forward(self, x):
        # x: (step, B, C, L)
        step, B, C, L = x.shape
        
        # 每次前向传播重新初始化膜电位
        membrane = torch.zeros(B, C, L, device=x.device)
        
        outputs = []
        for t in range(step):
            current_input = x[t]  # (B, C, L)
            
            # DCT频域处理
            x_freq = self.dct_transform(current_input)  # (B, C, freq_num)
            
            # 频域卷积
            x_freq = self.dct_conv(x_freq)  # (B, C//reduction, freq_num)
            x_freq = F.relu(x_freq)
            x_freq = self.freq_conv(x_freq)  # (B, C//reduction, freq_num)
            x_freq = F.relu(x_freq)
            x_freq = self.idct_conv(x_freq)  # (B, C, freq_num)
            
            # 逆DCT变换
            enhanced_input = self.idct_transform(x_freq, L)  # (B, C, L)
            
            # LIF神经元动力学
            membrane = self.leak.view(1, -1, 1) * membrane + enhanced_input
            
            # 脉冲发放
            spike_mask = (membrane > self.threshold.view(1, -1, 1)).float()
            output = spike_mask * membrane
            
            # 重置膜电位
            membrane = membrane * (1 - spike_mask)
            
            outputs.append(output)
        
        return torch.stack(outputs, dim=0)  # (step, B, C, L)

class SpikeBasicBlock1D(nn.Module):
    """
    Spike Basic Block adapted for 1D time series
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None, step=2, length=96, freq_num=32, reduction=16):
        super(SpikeBasicBlock1D, self).__init__()
        self.step = step
        
        # 第一个卷积
        self.conv1 = SpikePool1D(
            pool=nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            step=step
        )
        self.bn1 = myBatchNorm1d(bn=nn.BatchNorm1d(planes), step=step)
        self.relu1 = DctSpatialLIF1D(step=step, channel=planes, length=length//stride, freq_num=freq_num, reduction=reduction)
        
        # 第二个卷积
        self.conv2 = SpikePool1D(
            pool=nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            step=step
        )
        self.bn2 = myBatchNorm1d(bn=nn.BatchNorm1d(planes), step=step)
        
        self.downsample = downsample
        self.stride = stride
        
        if downsample is None:
            self.relu2 = DctSpatialLIF1D(step=step, channel=planes, length=length//stride, freq_num=freq_num, reduction=reduction)
        else:
            # 获取downsample输出通道数
            out_channels = planes
            self.relu2 = DctSpatialLIF1D(step=step, channel=out_channels, length=length//stride, freq_num=freq_num, reduction=reduction)

    def forward(self, s):
        temp, x = s  # (step, B, C, L) each
        residual = x
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = out + residual
        
        # 最终激活
        out1 = self.relu2(out.clone())
        
        return out, out1

class SpikeEncoder1D(nn.Module):
    """
    Spike-based encoder for time series
    """
    def __init__(self, configs):
        super(SpikeEncoder1D, self).__init__()
        self.step = 4  # 脉冲步数
        self.seq_len = configs.seq_len
        
        # 初始卷积
        self.conv1 = SpikePool1D(
            pool=nn.Conv1d(configs.enc_in, 64, kernel_size=7, stride=1, padding=3, bias=False),
            step=self.step
        )
        self.bn1 = myBatchNorm1d(bn=nn.BatchNorm1d(64), step=self.step)
        self.relu = DctSpatialLIF1D(step=self.step, channel=64, length=self.seq_len, freq_num=32, reduction=16)
        
        # 残差层
        self.layer1 = self._make_layer(64, 64, 2, stride=1, length=self.seq_len)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, length=self.seq_len)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, length=self.seq_len//2)
        
        # 全局平均池化
        self.avgpool = SpikePool1D(pool=nn.AdaptiveAvgPool1d(1), step=self.step)
        
        # 输出投影
        self.fc = SpikeConv1D(conv=nn.Linear(256, configs.d_model), step=self.step)
        
        self.inplanes = 64

    def _make_layer(self, inplanes, planes, blocks, stride=1, length=96):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                SpikePool1D(
                    pool=nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                    step=self.step
                ),
                myBatchNorm1d(bn=nn.BatchNorm1d(planes), step=self.step)
            )
        
        layers = []
        layers.append(SpikeBasicBlock1D(inplanes, planes, stride, downsample, step=self.step, length=length))
        for _ in range(1, blocks):
            layers.append(SpikeBasicBlock1D(planes, planes, step=self.step, length=length//stride))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)
        
        # 脉冲编码
        x = self.conv1(x)  # (step, B, 64, L)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 残差层
        temp, x = self.layer1((x, x))
        temp, x = self.layer2((temp, x))
        temp, x = self.layer3((temp, x))
        
        # 全局池化
        x = self.avgpool(x)  # (step, B, 256, 1)
        
        # 时间平均
        x = x.mean(dim=0)  # (B, 256, 1)
        x = x.squeeze(-1)  # (B, 256)
        
        # 输出投影 - 简化处理
        proj_layer = nn.Linear(256, 512).to(x.device)
        x = proj_layer(x)  # (B, 512)
        
        return x.unsqueeze(1)  # (B, 1, 512)

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
    ZAAA: Enhanced AdaWaveNet with Spike Neural Network components
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
        
        # Spike-based encoder
        self.spike_encoder = SpikeEncoder1D(configs)
        
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

        # Enhanced Encoder with Spike components
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
        
        # Fusion layer for spike and transformer features
        self.fusion_layer = nn.Linear(configs.d_model * 2, configs.d_model)
        
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
        
        # Spike encoding
        spike_out = self.spike_encoder(x_enc)  # (B, 1, d_model)
        if len(spike_out.shape) == 4:  # 处理可能的4D输出
            spike_out = spike_out.mean(dim=0)  # 平均时间步
        spike_out = spike_out.expand(-1, enc_out.size(1), -1)  # (B, L, d_model)
        
        # Fusion of transformer and spike features
        fused_features = torch.cat([enc_out, spike_out], dim=-1)  # (B, L, 2*d_model)
        fused_features = self.fusion_layer(fused_features)  # (B, L, d_model)
        
        enc_out, attns = self.encoder(fused_features, attn_mask=None)
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
