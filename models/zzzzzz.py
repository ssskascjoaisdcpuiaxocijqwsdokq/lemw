import torch
import torch.nn as nn
from layers.Transformer_Encoder import Encoder, EncoderLayer
from layers.SWTAttention_Family import GeomAttention
from layers.Embed import DataEmbedding_inverted
from layers.LiftingScheme import LiftingScheme, InverseLiftingScheme


def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)


class YanccWaveletDecomposer(nn.Module):
    """
    使用Yancc中的lifting方案对SimpleTM的WaveletEmbedding进行替换，
    以保持channels优先 (Variate-first) 的张量格式。
    """

    def __init__(self, d_channel: int, d_model: int, kernel_size: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model ({d_model}) 必须为偶数以便lifting划分。")

        self.d_channel = d_channel
        self.d_model = d_model
        self.wavelet = LiftingScheme(
            d_channel, input_size=d_model, k_size=kernel_size, simple_lifting=True
        )
        self.norm_c = normalization(d_channel)
        self.norm_d = normalization(d_channel)
        self.expand_c = nn.Linear(d_model // 2, d_model)
        self.expand_d = nn.Linear(d_model // 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Variate, d_model]
        c, d = self.wavelet(x)
        c = self.expand_c(self.norm_c(c))
        d = self.expand_d(self.norm_d(d))
        # 返回 [B, Variate, levels(=2), d_model]
        return torch.stack([c, d], dim=2)


class YanccWaveletReconstructor(nn.Module):
    """
    将注意力后的结果映射回lifting域，再通过InverseLifting恢复。
    """

    def __init__(self, d_channel: int, d_model: int, kernel_size: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model ({d_model}) 必须为偶数以便lifting划分。")

        self.d_channel = d_channel
        self.d_model = d_model
        self.reduce_c = nn.Linear(d_model, d_model // 2)
        self.reduce_d = nn.Linear(d_model, d_model // 2)
        self.inverse_wavelet = InverseLiftingScheme(
            d_channel, input_size=d_model // 2, kernel_size=kernel_size
        )

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        # coeffs: [B, Variate, levels(=2), d_model]
        approx = self.reduce_c(coeffs[:, :, 0, :])
        if coeffs.size(2) > 1:
            detail = self.reduce_d(coeffs[:, :, 1, :])
        else:
            detail = torch.zeros_like(approx)
        recon = self.inverse_wavelet(approx, detail)
        return recon


class YanccGeomAttentionLayer(nn.Module):
    """
    几何注意力层，使用Yancc的lifting小波模块替换SimpleTM的SWT嵌入。
    """

    def __init__(
        self,
        attention,
        d_model: int,
        d_channel: int,
        kernel_size: int,
        geomattn_dropout: float = 0.1,
    ):
        super().__init__()
        self.inner_attention = attention
        self.decomposer = YanccWaveletDecomposer(d_channel, d_model, kernel_size)
        self.query_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout),
        )
        self.key_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout),
        )
        self.value_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout),
        )
        self.out_linear = nn.Linear(d_model, d_model)
        self.reconstructor = YanccWaveletReconstructor(d_channel, d_model, kernel_size)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        queries = self.decomposer(queries)
        keys = self.decomposer(keys)
        values = self.decomposer(values)

        B, V, Lvl, D = queries.shape
        L = V * Lvl

        queries = self.query_projection(queries.reshape(B, L, D)).unsqueeze(2)
        keys = self.key_projection(keys.reshape(B, L, D)).unsqueeze(2)
        values = self.value_projection(values.reshape(B, L, D)).unsqueeze(2)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
        )

        out = self.out_linear(out.squeeze(2)).reshape(B, V, Lvl, D)
        out = self.reconstructor(out)
        return out, attn


class Model(nn.Module):
    """
    SimpleTM骨架 + Yancc lifting小波模块。
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.geomattn_dropout = configs.geomattn_dropout
        self.alpha = configs.alpha
        self.kernel_size = configs.kernel_size
        self.lifting_kernel_size = getattr(configs, "lifting_kernel_size", 4)

        enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.enc_embedding = enc_embedding

        encoder = Encoder(
            [
                EncoderLayer(
                    YanccGeomAttentionLayer(
                        GeomAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                            alpha=self.alpha,
                        ),
                        configs.d_model,
                        d_channel=configs.dec_in,
                        kernel_size=self.lifting_kernel_size,
                        geomattn_dropout=self.geomattn_dropout,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.encoder = encoder

        projector = nn.Linear(configs.d_model, self.pred_len, bias=True)
        self.projector = projector

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, None, None, None)
        if self.output_attention:
            return dec_out, attns
        return dec_out
