import torch
from .yecamrdu import Model as BaseModel


class Model(BaseModel):
    """
    ZCVX: yecamrdu variant that shuffles time steps before processing
    (destroys temporal order by random permutation along the sequence length).
    """

    def _shuffle_time(self, x, x_mark=None):
        # x: (B, L, C)
        perm = torch.randperm(x.shape[1], device=x.device)
        x_shuffled = x[:, perm, :]
        x_mark_shuffled = x_mark[:, perm, :] if x_mark is not None else None
        return x_shuffled, x_mark_shuffled

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x_enc, x_mark_enc = self._shuffle_time(x_enc, x_mark_enc)
        if x_dec is not None:
            x_dec, x_mark_dec = self._shuffle_time(x_dec, x_mark_dec)
        return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
