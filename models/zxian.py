import torch
import torch.nn as nn
from layers.Invertible import RevIN
from .yecamrdu import Model as BaseModel
from einops import rearrange


class LinearTrend(nn.Module):
    """Pure linear trend extractor per channel with RevIN."""
    def __init__(self, seq_len, pred_len, enc_in, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.revin = RevIN(enc_in)
        self.linear = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        x_norm = self.revin(x, 'norm')               # (B, L, C)
        x_norm = rearrange(x_norm, 'b l c -> (b c) l')
        out = self.linear(x_norm)                    # (B*C, pred_len)
        out = self.dropout(out)
        out = rearrange(out, '(b c) p -> b p c', b=B, c=C)
        out = self.revin(out, 'denorm')
        return out


class Model(BaseModel):
    """
    ZXIAN: yecamrdu with the trend extractor replaced by a pure linear layer.
    """
    def __init__(self, configs):
        super().__init__(configs)
        # Replace trend_extractor with a simple linear version
        self.trend_extractor = LinearTrend(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            enc_in=configs.enc_in,
            dropout=configs.dropout
        )
