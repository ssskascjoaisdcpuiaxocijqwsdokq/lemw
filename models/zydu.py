import torch
import torch.nn as nn
from .yecamrdu import Model as BaseModel


class ECABlock1D_MRFP_ChannelAlpha(nn.Module):
    """
    SE-MRFP with channel-wise gating (alpha is a per-channel vector).
    """
    def __init__(self, channel, kernel_size=3, dropout=0.1, mrfp_ratio=2.0, reduction=4):
        super().__init__()
        # MRFP multi-scale depthwise convs
        self.mrfp_conv3 = nn.Conv1d(channel, channel, kernel_size=3, padding=1, groups=channel)
        self.mrfp_conv5 = nn.Conv1d(channel, channel, kernel_size=5, padding=2, groups=channel)
        self.mrfp_conv7 = nn.Conv1d(channel, channel, kernel_size=7, padding=3, groups=channel)
        self.mrfp_fusion = nn.Sequential(
            nn.Conv1d(channel * 3, channel, kernel_size=1),
            nn.BatchNorm1d(channel),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # SE attention
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        reduced_dim = max(channel // reduction, 4)
        self.se_mlp = nn.Sequential(
            nn.Linear(channel, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, channel, bias=False),
            nn.Tanh(),
        )
        # Channel-wise gating (initialized to zero -> starts as identity)
        self.alpha = nn.Parameter(torch.zeros(1, channel, 1))

    def forward(self, x):
        # x: (B, C, L)
        residual = x
        B, C, L = x.shape
        # MRFP
        feat3 = self.mrfp_conv3(x)
        feat5 = self.mrfp_conv5(x)
        feat7 = self.mrfp_conv7(x)
        mrfp_feat = torch.cat([feat3, feat5, feat7], dim=1)
        mrfp_feat = self.mrfp_fusion(mrfp_feat)
        # SE attention
        y = self.avg_pool(mrfp_feat).view(B, C)
        attn_scale = self.se_mlp(y).unsqueeze(-1)  # (B, C, 1)
        enhanced_feat = mrfp_feat * (1.0 + attn_scale)
        # Channel-wise gated fusion
        out = residual + self.alpha * enhanced_feat
        return out


class Model(BaseModel):
    """
    ZYDU: yecamrdu variant using channel-wise gated SE-MRFP block.
    """
    def __init__(self, configs):
        super().__init__(configs)
        self.eca_mrfp_block = ECABlock1D_MRFP_ChannelAlpha(
            channel=configs.enc_in,
            dropout=configs.dropout
        )
