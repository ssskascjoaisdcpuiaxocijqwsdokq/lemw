import torch
from .yecamrdu import Model as BaseModel


class Model(BaseModel):
    """
    ZTLUAN: 在 yecamrdu 基础上，前向时随机打乱通道顺序（encoder/decoder 输入一致打乱），
    其余结构保持不变。
    """
    def _shuffle_channels(self, x_enc, x_dec=None):
        # x_enc: (B, L, C)
        perm = torch.randperm(x_enc.shape[-1], device=x_enc.device)
        x_enc = x_enc[:, :, perm]
        if x_dec is not None:
            x_dec = x_dec[:, :, perm]
        return x_enc, x_dec

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 在进入父类逻辑前打乱通道
        x_enc, x_dec = self._shuffle_channels(x_enc, x_dec)
        return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
