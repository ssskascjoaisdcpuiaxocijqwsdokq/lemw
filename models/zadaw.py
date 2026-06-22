import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.LiftingScheme import LiftingScheme, InverseLiftingScheme
from layers.Invertible import RevIN
from torch.nn import init
from einops import rearrange

def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)

class MultiDWConv1D(nn.Module):
    """
    将VITComer的MultiDWConv适配为1D时序数据版本
    多感受野深度可分离卷积，适配时序预测任务
    """
    def __init__(self, dim=768):
        super().__init__()
        self.full_dim = dim
        self.half_dim = dim // 2
        self.other_half = dim - self.half_dim

        # 1D深度可分离卷积，使用不同的kernel size模拟多感受野
        self.dwconv1 = nn.Conv1d(self.half_dim, self.half_dim, 3, 1, 1, bias=True, groups=self.half_dim)
        self.dwconv2 = nn.Conv1d(self.other_half, self.other_half, 5, 1, 2, bias=True, groups=self.other_half)

        self.dwconv3 = nn.Conv1d(self.half_dim, self.half_dim, 7, 1, 3, bias=True, groups=self.half_dim)
        self.dwconv4 = nn.Conv1d(self.other_half, self.other_half, 9, 1, 4, bias=True, groups=self.other_half)

        self.dwconv5 = nn.Conv1d(self.half_dim, self.half_dim, 11, 1, 5, bias=True, groups=self.half_dim)
        self.dwconv6 = nn.Conv1d(self.other_half, self.other_half, 13, 1, 6, bias=True, groups=self.other_half)

        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.full_dim)

        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.full_dim)

        self.act3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.full_dim)

        # 可学习的多尺度融合权重（通过softmax规范化）
        self.fusion_logits = nn.Parameter(torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32))

    def forward(self, x, seq_len):
        """
        x: (B, L, C) - 时序特征
        seq_len: 序列长度，用于多尺度分割
        """
        B, L, C = x.shape
        
        # 将时序数据分割为3个不同的尺度（模拟VITComer的多尺度处理）
        # 长尺度：完整序列
        # 中尺度：序列的中间部分
        # 短尺度：序列的核心部分
        scale1_len = L
        scale2_len = L // 2
        scale3_len = L // 4
        
        # 提取不同尺度的特征
        x1 = x  # 完整序列 (B, L, C)
        x2 = x[:, L//4:L//4+scale2_len, :]  # 中间部分 (B, L//2, C)
        x3 = x[:, L//2-scale3_len//2:L//2+scale3_len//2, :]  # 核心部分 (B, L//4, C)
        
        # 转换为卷积格式 (B, C, L)
        x1 = x1.transpose(1, 2)  # (B, C, L)
        x2 = x2.transpose(1, 2)  # (B, C, L//2)
        x3 = x3.transpose(1, 2)  # (B, C, L//4)

        # 第一个尺度：应用不同感受野的卷积
        x11, x12 = x1[:, :self.half_dim, :], x1[:, self.half_dim:, :]
        x11 = self.dwconv1(x11)  # 3x1卷积
        x12 = self.dwconv2(x12)  # 5x1卷积
        x1 = torch.cat([x11, x12], dim=1)
        x1 = self.act1(self.bn1(x1)).transpose(1, 2)  # (B, L, C)

        # 第二个尺度：应用不同感受野的卷积
        x21, x22 = x2[:, :self.half_dim, :], x2[:, self.half_dim:, :]
        x21 = self.dwconv3(x21)  # 7x1卷积
        x22 = self.dwconv4(x22)  # 9x1卷积
        x2 = torch.cat([x21, x22], dim=1)
        x2 = self.act2(self.bn2(x2)).transpose(1, 2)  # (B, L//2, C)

        # 第三个尺度：应用不同感受野的卷积
        x31, x32 = x3[:, :self.half_dim, :], x3[:, self.half_dim:, :]
        x31 = self.dwconv5(x31)  # 11x1卷积
        x32 = self.dwconv6(x32)  # 13x1卷积
        x3 = torch.cat([x31, x32], dim=1)
        x3 = self.act3(self.bn3(x3)).transpose(1, 2)  # (B, L//4, C)

        # 将不同尺度的特征插值到统一长度并融合
        x2_interp = F.interpolate(x2.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
        x3_interp = F.interpolate(x3.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
        
        # 自适应加权融合多尺度特征
        weights = F.softmax(self.fusion_logits, dim=0)  # (3,)
        x = weights[0] * x1 + weights[1] * x2_interp + weights[2] * x3_interp  # (B, L, C)
        
        return x

class MRFP1D(nn.Module):
    """
    将VITComer的MRFP适配为1D时序版本
    Multi-Receptive Field Processor for 1D Time Series
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = MultiDWConv1D(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, seq_len):
        """
        x: (B, L, C) - 时序特征
        seq_len: 序列长度
        """
        x = self.fc1(x)  # (B, L, C) -> (B, L, hidden_features)
        x = self.dwconv(x, seq_len)  # 多感受野处理
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  # (B, L, hidden_features) -> (B, L, C)
        x = self.drop(x)
        return x

class ECAAttention1D(nn.Module):
    """
    ECA注意力机制适配1D时序数据版本
    Efficient Channel Attention for 1D Time Series
    """
    def __init__(self, channel=512, kernel_size=3, dropout=0.1):
        super().__init__()
        self.channel = channel
        self.kernel_size = kernel_size
        
        # 全局平均池化 - 适配1D数据 (B, C, L) -> (B, C, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # 1D卷积建模通道间相关性
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        
        # 批归一化和Dropout提升稳定性
        self.bn = nn.BatchNorm1d(1)
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        self.sigmoid = nn.Sigmoid()
        
        # 可学习的缩放参数
        self.scale = nn.Parameter(torch.ones(1))
        
        self.init_weights()

    def init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        init.constant_(self.scale, 0.1)

    def forward(self, x):
        """
        前向传播
        输入: x (B, C, L) - 批次大小, 通道数, 序列长度
        输出: (B, C, L) - 加权后的特征
        """
        B, C, L = x.size()
        residual = x
        
        # 全局平均池化: (B, C, L) -> (B, C, 1)
        y = self.gap(x)
        
        # 重塑为1D卷积输入格式: (B, C, 1) -> (B, 1, C)
        y = y.permute(0, 2, 1)
        
        # 1D卷积建模通道相关性: (B, 1, C) -> (B, 1, C)
        y = self.conv(y)
        y = self.bn(y)
        y = self.dropout(y)
        
        # 生成注意力权重: (B, 1, C)
        y = self.sigmoid(y)
        
        # 重塑回原始格式: (B, 1, C) -> (B, C, 1)
        y = y.permute(0, 2, 1)
        
        # 扩展到原始维度并应用注意力权重
        attention_weights = y.expand(-1, -1, L)
        
        # 加权特征融合
        enhanced_x = x * attention_weights
        
        # 残差连接和可学习缩放
        output = residual + enhanced_x * self.scale
        
        return output

class ECABlock1D_MRFP(nn.Module):
    """
    融合ECA注意力和MRFP多感受野处理的增强块
    """
    def __init__(self, channel=512, kernel_size=3, dropout=0.1, mrfp_ratio=2.0):
        super().__init__()
        self.channel = channel
        
        # 预归一化
        self.pre_norm = nn.LayerNorm(channel)
        
        # ECA注意力
        self.eca_attention = ECAAttention1D(channel=channel, kernel_size=kernel_size, dropout=dropout)
        
        # MRFP多感受野处理器
        self.mrfp = MRFP1D(
            in_features=channel,
            hidden_features=int(channel * mrfp_ratio),
            drop=dropout
        )
        
        # 特征增强网络
        self.feature_enhance = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(channel),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channel, channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(channel)
        )
        
        # 可学习的残差和注意力缩放参数
        self.residual_scale = nn.Parameter(torch.ones(1))
        self.attention_scale = nn.Parameter(torch.ones(1))
        self.mrfp_scale = nn.Parameter(torch.ones(1))
        
        self.init_weights()

    def init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        init.constant_(self.residual_scale, 0.3)
        init.constant_(self.attention_scale, 0.4)
        init.constant_(self.mrfp_scale, 0.3)

    def forward(self, x):
        """
        前向传播
        输入: x (B, C, L)
        输出: (B, C, L)
        """
        B, C, L = x.size()
        residual = x
        
        # 预归一化: (B, C, L) -> (B, L, C) -> LayerNorm -> (B, C, L)
        x_norm = self.pre_norm(x.transpose(1, 2)).transpose(1, 2)
        
        # ECA注意力
        attention_out = self.eca_attention(x_norm)
        
        # MRFP多感受野处理: (B, C, L) -> (B, L, C) -> MRFP -> (B, C, L)
        mrfp_input = attention_out.transpose(1, 2)
        mrfp_out = self.mrfp(mrfp_input, L).transpose(1, 2)
        
        # 特征增强
        enhanced_out = self.feature_enhance(mrfp_out)
        
        # 多分支残差连接和缩放
        output = (residual * self.residual_scale + 
                 attention_out * self.attention_scale + 
                 enhanced_out * self.mrfp_scale)
        
        return output

class DUETLinearExtractor(nn.Module):
    """
    基于DUET模型的线性提取器，用于趋势分量处理
    简化版本，专注于趋势预测
    """
    def __init__(self, seq_len, pred_len, enc_in, d_model=512, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        
        # RevIN归一化
        self.revin = RevIN(enc_in)
        
        # 线性投影层：将时序长度映射到模型维度
        self.temporal_projection = nn.Linear(seq_len, d_model)
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, pred_len),
            nn.Dropout(dropout)
        )
        
        # 权重初始化
        self.init_weights()
    
    def init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        输入: x (B, L, C) - 趋势分量
        输出: (B, pred_len, C) - 趋势预测
        """
        B, L, C = x.shape
        
        # RevIN归一化
        x_norm = self.revin(x, 'norm')
        
        # 通道独立处理: (B, L, C) -> (B*C, L, 1)
        x_reshaped = rearrange(x_norm, 'b l c -> (b c) l 1')
        
        # 时序投影: (B*C, L, 1) -> (B*C, d_model, 1)
        temporal_features = self.temporal_projection(x_reshaped.squeeze(-1))  # (B*C, d_model)
        
        # 特征提取
        extracted_features = self.feature_extractor(temporal_features)  # (B*C, d_model)
        
        # 预测
        predictions = self.prediction_head(extracted_features)  # (B*C, pred_len)
        
        # 重塑回原始格式: (B*C, pred_len) -> (B, pred_len, C)
        predictions = rearrange(predictions, '(b c) p -> b p c', b=B, c=C)
        
        # RevIN反归一化
        predictions = self.revin(predictions, 'denorm')
        
        return predictions

class ClusteredTrendAdapter(nn.Module):
    """对趋势预测进行簇特定的细化线性变换"""
    def __init__(self, n_clusters, pred_len, enc_in):
        super().__init__()
        self.enc_in = enc_in
        self.adapters = nn.ModuleDict({
            str(cluster_id): nn.Linear(pred_len, pred_len) for cluster_id in range(n_clusters)
        })

    def forward(self, trend_pred, clusters):
        """
        trend_pred: (B, pred_len, C)
        clusters: (C,) 每个通道的簇编号
        """
        assert trend_pred.size(2) == self.enc_in
        assert clusters.numel() == self.enc_in

        refined = []
        for ch in range(self.enc_in):
            cluster_id = str(int(clusters[ch].item()))
            channel_trend = trend_pred[:, :, ch]  # (B, pred_len)
            refined_channel = self.adapters[cluster_id](channel_trend).unsqueeze(-1)
            refined.append(refined_channel)
        return torch.cat(refined, dim=-1)

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

class moving_avg_imputation(nn.Module):
    """
    Moving average block modified to ignore zeros in the moving window.
    """
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
        weight = torch.ones((1, num_channels, self.kernel_size), device=x_padded.device)
        window_sum = torch.nn.functional.conv1d(x_padded, 
                                                weight=weight,
                                                stride=self.stride)

        # Count non-zero elements in each window
        window_count = torch.nn.functional.conv1d(non_zero_mask.float(), 
                                                  weight=weight,
                                                  stride=self.stride)

        # Avoid division by zero; set count to 1 where there are no non-zero elements
        window_count = torch.clamp(window_count, min=1)

        # Compute the moving average
        moving_avg = window_sum / window_count
        return moving_avg

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
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
    
class Model(nn.Module):
    """
    YECAMRDU模型: YECA + VITComer MRFP + DUET线性提取器
    融合ECA注意力机制、多感受野特征处理和DUET趋势提取
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
        self.debug = getattr(configs, 'debug', False)
        default_log = f"./logs/{getattr(configs, 'model_id', 'zadaw')}_debug.log"
        self.log_file = getattr(configs, 'log_file', default_log)
        self.log_limit = getattr(configs, 'log_limit', 50)
        self.trend_scale = getattr(configs, 'trend_scale', 0.5)
        self._current_epoch = 0
        self._log_count = 0
        self._logged_tags = set()
        self.series_decomp = series_decomp(imputation = self.task_name=='imputation')
        self.rev_seasonal = RevIN(configs.enc_in)
        
        # DUET线性提取器替代聚类线性层
        self.trend_extractor = DUETLinearExtractor(
            seq_len=self.seq_len,
            pred_len=configs.pred_len,
            enc_in=configs.enc_in,
            d_model=getattr(configs, 'duet_d_model', 512),
            dropout=configs.dropout
        )
        self.trend_adapter = ClusteredTrendAdapter(
            n_clusters=configs.n_clusters,
            pred_len=configs.pred_len,
            enc_in=configs.enc_in
        )
        self.trend_norm = nn.LayerNorm(self.pred_len)
        
        # 融合ECA和MRFP的增强注意力模块
        self.eca_mrfp_block = ECABlock1D_MRFP(
            channel=configs.enc_in, 
            kernel_size=getattr(configs, 'eca_kernel_size', 3),
            dropout=configs.dropout,
            mrfp_ratio=getattr(configs, 'mrfp_ratio', 2.0)
        )

        if self.debug:
            print("[zadaw] debug mode enabled")
        
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
        # 低秩解码后增加归一化以稳定尺度
        proj_dim = self.pred_len // (2 ** configs.lifting_levels) if self.task_name == "super_resolution" else self.seq_len // (2 ** configs.lifting_levels)
        self.lowrank_norm = nn.LayerNorm(proj_dim)
        nn.init.xavier_uniform_(self.lowrank_projection.weight, gain=0.5)
        if self.lowrank_projection.bias is not None:
            nn.init.zeros_(self.lowrank_projection.bias)

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
        self._log("input_x_enc", x_enc)
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        self._log("after_series_decomp_seasonal", x)
        self._log("after_series_decomp_trend", moving_mean)
        
        # 应用融合ECA和MRFP的增强注意力机制
        x = self.eca_mrfp_block(x)
        self._log("after_eca_mrfp", x)
        
        # Normalization from Non-stationary Transformer
        x_enc = x.permute(0,2,1)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev  # 避免in-place操作
        _, _, N = x_enc.shape

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
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        x_dec = self.lowrank_norm(x_dec)
        self._log("encoder_output", enc_out)
        self._log("lowrank_projection", x_dec)
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
        
        # 使用簇特定线性层细化趋势分量
        trend_base = self.trend_extractor(moving_mean)
        if len(torch.unique(clusters)) > 1:
            trend_adapted = self.trend_adapter(trend_base, clusters)
        else:
            trend_adapted = trend_base
        trend_normed = self.trend_norm(trend_adapted.permute(0, 2, 1)).permute(0, 2, 1)
        moving_mean_out = trend_normed * self.trend_scale
        self._log("trend_out", moving_mean_out)
        
        dec_out = dec_out + moving_mean_out
        self._log("forecast_output", dec_out)
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask, clusters):
        self._log("input_x_enc", x_enc)
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        self._log("after_series_decomp_seasonal", x)
        self._log("after_series_decomp_trend", moving_mean)
        
        # 应用融合ECA和MRFP的增强注意力机制
        x = self.eca_mrfp_block(x)
        self._log("after_eca_mrfp", x)
        
        # Normalization from Non-stationary Transformer
        x_enc = x.permute(0,2,1)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        _, _, N = x_enc.shape

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
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        x_dec = self.lowrank_norm(x_dec)
        self._log("encoder_output", enc_out)
        self._log("lowrank_projection", x_dec)
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
        
        # 使用簇特定线性层细化趋势分量
        trend_base = self.trend_extractor(moving_mean)
        if len(torch.unique(clusters)) > 1:
            trend_adapted = self.trend_adapter(trend_base, clusters)
        else:
            trend_adapted = trend_base
        trend_normed = self.trend_norm(trend_adapted.permute(0, 2, 1)).permute(0, 2, 1)
        moving_mean_out = trend_normed * self.trend_scale
        self._log("trend_out", moving_mean_out)
        
        dec_out = dec_out + moving_mean_out
        self._log("imputation_output", dec_out)
        
        return dec_out

    def anomaly_detection(self, x_enc, clusters):
        self._log("input_x_enc", x_enc)
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        self._log("after_series_decomp_seasonal", x)
        self._log("after_series_decomp_trend", moving_mean)
        
        # 应用融合ECA和MRFP的增强注意力机制
        x = self.eca_mrfp_block(x)
        self._log("after_eca_mrfp", x)
        
        # Normalization from Non-stationary Transformer
        x_enc = x.permute(0,2,1)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        _, _, N = x_enc.shape

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
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        x_dec = self.lowrank_norm(x_dec)
        self._log("encoder_output", enc_out)
        self._log("lowrank_projection", x_dec)
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
        
        # 使用簇特定线性层细化趋势分量
        trend_base = self.trend_extractor(moving_mean)
        if len(torch.unique(clusters)) > 1:
            trend_adapted = self.trend_adapter(trend_base, clusters)
        else:
            trend_adapted = trend_base
        trend_normed = self.trend_norm(trend_adapted.permute(0, 2, 1)).permute(0, 2, 1)
        moving_mean_out = trend_normed * self.trend_scale
        self._log("trend_out", moving_mean_out)
        
        dec_out = dec_out + moving_mean_out
        self._log("anomaly_output", dec_out)
        
        return dec_out

    def super_resolution(self, x_enc, clusters):
        self._log("input_x_enc", x_enc)
        x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
        moving_mean = moving_mean.permute(0,2,1)
        self._log("after_series_decomp_seasonal", x)
        self._log("after_series_decomp_trend", moving_mean)
        
        # 应用融合ECA和MRFP的增强注意力机制
        x = self.eca_mrfp_block(x)
        self._log("after_eca_mrfp", x)
        
        # Normalization from Non-stationary Transformer
        x_enc = x.permute(0,2,1)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        _, _, N = x_enc.shape
        
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
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        self._log("encoder_output", enc_out)
        self._log("lowrank_projection", x_dec)
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
        
        # 使用簇特定线性层细化趋势分量
        trend_base = self.trend_extractor(moving_mean)
        if len(torch.unique(clusters)) > 1:
            trend_adapted = self.trend_adapter(trend_base, clusters)
        else:
            trend_adapted = trend_base
        trend_normed = self.trend_norm(trend_adapted.permute(0, 2, 1)).permute(0, 2, 1)
        moving_mean_out = trend_normed * self.trend_scale
        self._log("trend_out", moving_mean_out)
        
        dec_out = dec_out + moving_mean_out
        self._log("super_resolution_output", dec_out)
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, C = x_enc.shape
        x_cluster = x_enc.permute(2, 0, 1).reshape(C, B * L)
        # 动态聚类：每次前向都重新计算簇分配
        clusters = self.kmeans.fit_predict(x_cluster)
        if self.debug and self._log_count < self.log_limit:
            uniq, counts = torch.unique(clusters, return_counts=True)
            msg = f"clusters={clusters.tolist()} dist={list(zip(uniq.tolist(), counts.tolist()))}"
            self._log_text("kmeans", msg)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, clusters)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask, clusters)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, clusters)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == 'super_resolution':
            dec_out = self.super_resolution(x_enc, clusters)
            return dec_out
        return None

    def set_epoch(self, epoch: int):
        """由训练循环在每个epoch开始时调用，用于限制日志次数"""
        self._current_epoch = epoch
        self._reset_log_state()

    def _reset_log_state(self):
        self._log_count = 0
        self._logged_tags = set()
        self._log_enabled = True

    def _log(self, name, tensor):
        if not self.debug or not self._log_enabled:
            return
        if self._log_count >= self.log_limit or name in self._logged_tags:
            return
        with torch.no_grad():
            try:
                mean = tensor.mean().item()
                std = tensor.std().item()
                min_v = tensor.min().item()
                max_v = tensor.max().item()
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True) if os.path.dirname(self.log_file) else None
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"[epoch {self._current_epoch}][zadaw][{name}] shape={tuple(tensor.shape)} "
                            f"mean={mean:.4f} std={std:.4f} min={min_v:.4f} max={max_v:.4f}\n")
                self._log_count += 1
                self._logged_tags.add(name)
            except Exception as e:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"[epoch {self._current_epoch}][zadaw][{name}] log failed: {e}\n")

    def _log_text(self, name: str, msg: str):
        if not self.debug or not self._log_enabled or self._log_count >= self.log_limit or name in self._logged_tags:
            return
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True) if os.path.dirname(self.log_file) else None
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[epoch {self._current_epoch}][zadaw][{name}] {msg}\n")
        self._log_count += 1
        self._logged_tags.add(name)
