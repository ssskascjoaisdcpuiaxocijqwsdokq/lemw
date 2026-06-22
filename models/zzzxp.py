import torch
import torch.nn as nn
from .yecamrdu import Model as BaseModel


class Channel_Mixer_Attention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mixer = nn.Conv1d(channel, channel, kernel_size=1)
        self.act = nn.Tanh()

    def forward(self, x):
        y = self.avg_pool(x)          # [B, C, 1]
        y = self.mixer(y)             # [B, C, 1]
        scale = self.act(y)           # [-1, 1]
        return x * (1.0 + scale)


class Sorted_ECA_MRFP_Block(nn.Module):
    def __init__(self, channel, dropout=0.1):
        super().__init__()
        # MRFP 卷积
        self.mrfp_conv3 = nn.Conv1d(channel, channel, kernel_size=3, padding=1, groups=channel)
        self.mrfp_conv5 = nn.Conv1d(channel, channel, kernel_size=5, padding=2, groups=channel)
        self.mrfp_conv7 = nn.Conv1d(channel, channel, kernel_size=7, padding=3, groups=channel)
        self.mrfp_fusion = nn.Sequential(
            nn.Conv1d(channel * 3, channel, kernel_size=1),
            nn.BatchNorm1d(channel),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Channel Mixer 注意力替换 ECA/SE
        self.se_attn = Channel_Mixer_Attention(channel)
        # 门控
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [B, C, L]
        residual = x
        # MRFP
        feat3 = self.mrfp_conv3(x)
        feat5 = self.mrfp_conv5(x)
        feat7 = self.mrfp_conv7(x)
        mrfp_feat = self.mrfp_fusion(torch.cat([feat3, feat5, feat7], dim=1))
        # SE 注意力
        enhanced_feat = self.se_attn(mrfp_feat)
        out = residual + self.alpha * enhanced_feat
        return out


class Model(BaseModel):
    """
    ZZZXP: yecamrdu with Sorted_ECA_MRFP_Block replacing the original ECA+MRFP block.
    """
    def __init__(self, configs):
        super().__init__(configs)
        self.eca_mrfp_block = Sorted_ECA_MRFP_Block(
            channel=configs.enc_in,
            dropout=configs.dropout
        )
