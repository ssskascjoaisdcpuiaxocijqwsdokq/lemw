import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.Invertible import RevIN
from .yecamrdu import series_decomp, DUETLinearExtractor, Adaptive_ECA_MRFP


class Model(nn.Module):
    """
    ZBOWU: yecamrdu 去掉小波模块的简化版，仅保留 ECA+MRFP（自适应门控）和 DUET 趋势头。
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'super_resolution':
            self.seq_len = self.seq_len // configs.sr_ratio
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.series_decomp = series_decomp(imputation=self.task_name == 'imputation')
        self.rev_seasonal = RevIN(configs.enc_in)

        # 趋势提取：DUET 线性头（通道独立）
        self.trend_extractor = DUETLinearExtractor(
            seq_len=self.seq_len,
            pred_len=configs.pred_len,
            enc_in=configs.enc_in,
            d_model=getattr(configs, 'duet_d_model', 512),
            dropout=configs.dropout
        )

        # 自适应门控的 ECA+MRFP（默认 gate=0，初始恒等）
        self.eca_mrfp_block = Adaptive_ECA_MRFP(
            channel=configs.enc_in,
            kernel_size=getattr(configs, 'eca_kernel_size', 3),
            dropout=configs.dropout,
            mrfp_ratio=getattr(configs, 'mrfp_ratio', 2.0),
            gate_init=getattr(configs, 'adaptive_gate_init', 0.0)
        )

        # Embedding （不做小波下采样，直接基于 seq_len）
        self.enc_embedding = DataEmbedding_inverted(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # Transformer 编码器
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
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # 投影层：直接将 d_model 投到目标长度
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, self.seq_len, bias=True)
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, self.seq_len, bias=True)
        elif self.task_name == 'super_resolution':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    # ====== 辅助 ======
    def _norm(self, x):
        x_enc = x.permute(0, 2, 1)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        return x_enc, means, stdev

    def _denorm(self, dec_out, means, stdev):
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    # ====== 任务分支 ======
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x, moving_mean = self.series_decomp(x_enc.permute(0, 2, 1))
        moving_mean = moving_mean.permute(0, 2, 1)
        x = self.eca_mrfp_block(x)

        x_enc, means, stdev = self._norm(x)
        _, _, N = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denorm(dec_out, means, stdev)

        moving_mean_out = self.trend_extractor(moving_mean)
        dec_out = dec_out + moving_mean_out
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x, moving_mean = self.series_decomp(x_enc.permute(0, 2, 1))
        moving_mean = moving_mean.permute(0, 2, 1)
        x = self.eca_mrfp_block(x)

        x_enc, means, stdev = self._norm(x)
        _, _, N = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denorm(dec_out, means, stdev)

        moving_mean_out = self.trend_extractor(moving_mean)
        dec_out = dec_out + moving_mean_out
        return dec_out

    def anomaly_detection(self, x_enc):
        x, moving_mean = self.series_decomp(x_enc.permute(0, 2, 1))
        moving_mean = moving_mean.permute(0, 2, 1)
        x = self.eca_mrfp_block(x)

        x_enc, means, stdev = self._norm(x)
        _, _, N = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denorm(dec_out, means, stdev)

        moving_mean_out = self.trend_extractor(moving_mean)
        dec_out = dec_out + moving_mean_out
        return dec_out

    def super_resolution(self, x_enc):
        x, moving_mean = self.series_decomp(x_enc.permute(0, 2, 1))
        moving_mean = moving_mean.permute(0, 2, 1)
        x = self.eca_mrfp_block(x)

        x_enc, means, stdev = self._norm(x)
        _, _, N = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denorm(dec_out, means, stdev)

        moving_mean_out = self.trend_extractor(moving_mean)
        dec_out = dec_out + moving_mean_out
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        if self.task_name == 'super_resolution':
            return self.super_resolution(x_enc)
        return None
