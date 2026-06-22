import math
import numpy as np
import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.Invertible import RevIN
# 复用 yecamrdu 中已有的模块，保持主体结构一致
from .yecamrdu import ECABlock1D_MRFP, series_decomp, AdpWaveletBlock, InverseAdpWaveletBlock


class TrendClusteredLinear(nn.Module):
    """按通道均值/方差聚类后的趋势线性层，支持每个簇独立线性映射。"""
    def __init__(self, n_clusters, seq_len, pred_len, enc_in, dropout=0.1):
        super().__init__()
        self.n_clusters = n_clusters
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.linears = nn.ModuleDict({
            str(cid): nn.Sequential(
                nn.Linear(seq_len, seq_len * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(seq_len * 2, pred_len)
            ) for cid in range(n_clusters)
        })
        # 对趋势分量做简单归一化/反归一化
        self.revin = RevIN(enc_in)

    def forward(self, x, clusters):
        # x: (B, L, C), clusters: (C,)
        B, L, C = x.shape
        x = self.revin(x, 'norm')
        outputs = []
        for ch in range(C):
            cid = str(int(clusters[ch].item()))
            ch_data = x[:, :, ch]
            ch_out = self.linears[cid](ch_data)
            outputs.append(ch_out.unsqueeze(-1))
        out = torch.cat(outputs, dim=-1)  # (B, pred_len, C)
        out = self.revin(out, 'denorm')
        return out


class Model(nn.Module):
    """
    ZAWN: 在 YECAMRDU 基础上加入均值/方差通道聚类的趋势线性层。
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'super_resolution':
            self.seq_len = self.seq_len // configs.sr_ratio
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.n_clusters = configs.n_clusters
        self.clusters = None

        self.series_decomp = series_decomp(imputation=self.task_name == 'imputation')
        self.rev_seasonal = RevIN(configs.enc_in)
        self.rev_trend = RevIN(configs.enc_in)

        # 均值/方差聚类趋势层
        self.trend_extractor = TrendClusteredLinear(
            n_clusters=self.n_clusters,
            seq_len=self.seq_len,
            pred_len=configs.pred_len,
            enc_in=configs.enc_in,
            dropout=configs.dropout
        )

        # ECA + MRFP
        self.eca_mrfp_block = ECABlock1D_MRFP(
            channel=configs.enc_in,
            kernel_size=getattr(configs, 'eca_kernel_size', 3),
            dropout=configs.dropout,
            mrfp_ratio=getattr(configs, 'mrfp_ratio', 2.0)
        )

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len // (2 ** configs.lifting_levels),
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # 小波编码/解码层级
        self.encoder_levels = nn.ModuleList()
        self.linear_levels = nn.ModuleList()
        self.coef_linear_levels = nn.ModuleList()
        self.coef_dec_levels = nn.ModuleList()
        input_size = self.seq_len
        expand_ratio = configs.sr_ratio if self.task_name == "super_resolution" else 1
        for i in range(configs.lifting_levels):
            self.encoder_levels.add_module('encoder_level_' + str(i), AdpWaveletBlock(configs, input_size))
            input_size = input_size // 2
            self.linear_levels.add_module('linear_level_' + str(i), nn.Sequential(nn.Linear(input_size, input_size * expand_ratio)))
            self.coef_linear_levels.add_module('linear_level_' + str(i), nn.Sequential(nn.Linear(input_size, input_size * expand_ratio)))
            self.coef_dec_levels.add_module('linear_level_' + str(i), nn.Sequential(nn.Linear(input_size, input_size * expand_ratio)))

        self.input_size = input_size

        # 解码器
        self.decoder_levels = nn.ModuleList()
        for i in range(configs.lifting_levels - 1, -1, -1):
            self.decoder_levels.add_module('decoder_level_' + str(i), InverseAdpWaveletBlock(configs, input_size=input_size))
            input_size *= 2

        if self.task_name == "super_resolution":
            self.lowrank_projection = nn.Linear(configs.d_model, self.pred_len // (2 ** configs.lifting_levels), bias=True)
        else:
            self.lowrank_projection = nn.Linear(configs.d_model, self.seq_len // (2 ** configs.lifting_levels), bias=True)

        # 编码器
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model,
                        configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # 投影层
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(self.seq_len, configs.pred_len, bias=True)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(self.seq_len, self.seq_len, bias=True)
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(self.seq_len, self.seq_len, bias=True)
        elif self.task_name == 'super_resolution':
            self.projection = nn.Linear(configs.pred_len, configs.pred_len, bias=True)

    # ======== 聚类工具 ========
    def calc_meanvar_clusters(self, data, n_clusters):
        # data shape: (B, L, C)
        with torch.no_grad():
            vals = data.detach().cpu()
            means = vals.mean(dim=(0, 1))  # (C,)
            stds = vals.std(dim=(0, 1))    # (C,)
            features = torch.stack([means, stds], dim=1).numpy()  # (C,2)
            norm_feats = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0) + 1e-8)
            scores = norm_feats.sum(axis=1)
            order = scores.argsort()
            clusters = np.zeros_like(scores, dtype=np.int64)
            split_size = max(1, len(scores) // n_clusters)
            for i in range(n_clusters):
                if i == n_clusters - 1:
                    clusters[order[i * split_size:]] = i
                else:
                    clusters[order[i * split_size:(i + 1) * split_size]] = i
            return torch.from_numpy(clusters).to(data.device)

    # ======== 公共流程 ========
    def _encode(self, x):
        encoded_coefficients, x_embedding_levels, coef_embedding_levels = [], [], []
        for l, l_linear, c_linear in zip(self.encoder_levels, self.linear_levels, self.coef_linear_levels):
            x, r, details = l(x)
            encoded_coefficients.append(details)
            coef_embedding_levels.append(c_linear(details))
            x_embedding_levels.append(l_linear(x))
        return x, encoded_coefficients, x_embedding_levels, coef_embedding_levels

    def _decode(self, x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels):
        x_enc = x_enc.permute(0, 2, 1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        for dec, x_emb_level, coef_emb_level, c_linear in zip(
            self.decoder_levels, x_embedding_levels[::-1], coef_embedding_levels[::-1], self.coef_dec_levels[::-1]
        ):
            details = encoded_coefficients.pop()
            details = coef_emb_level + c_linear(details)
            x_dec = x_dec + x_emb_level
            x_dec = dec(x_dec, details)
        return x_dec

    def _denorm(self, dec_out, means, stdev):
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def _norm(self, x):
        x_enc = x.permute(0, 2, 1)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        return x_enc, means, stdev

    # ======== 任务分支 ========
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x, moving_mean = self.series_decomp(x_enc.permute(0, 2, 1))
        moving_mean = moving_mean.permute(0, 2, 1)
        x = self.eca_mrfp_block(x)
        x_enc, means, stdev = self._norm(x)
        N = x_enc.shape[1]  # original sequence length
        # _encode expects (B, C, L)
        x_enc = x_enc.permute(0, 2, 1)
        x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels = self._encode(x_enc)
        x_dec = self._decode(x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denorm(dec_out, means, stdev)

        if self.clusters is None or len(self.clusters) != moving_mean.shape[-1]:
            self.clusters = self.calc_meanvar_clusters(moving_mean, self.n_clusters)
        moving_mean_out = self.trend_extractor(moving_mean, self.clusters)
        dec_out = dec_out + moving_mean_out
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x, moving_mean = self.series_decomp(x_enc.permute(0, 2, 1))
        moving_mean = moving_mean.permute(0, 2, 1)
        x = self.eca_mrfp_block(x)
        x_enc, means, stdev = self._norm(x)
        N = x_enc.shape[1]
        x_enc = x_enc.permute(0, 2, 1)
        x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels = self._encode(x_enc)
        x_dec = self._decode(x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denorm(dec_out, means, stdev)

        if self.clusters is None or len(self.clusters) != moving_mean.shape[-1]:
            self.clusters = self.calc_meanvar_clusters(moving_mean, self.n_clusters)
        moving_mean_out = self.trend_extractor(moving_mean, self.clusters)
        dec_out = dec_out + moving_mean_out
        return dec_out

    def anomaly_detection(self, x_enc):
        x, moving_mean = self.series_decomp(x_enc.permute(0, 2, 1))
        moving_mean = moving_mean.permute(0, 2, 1)
        x = self.eca_mrfp_block(x)
        x_enc, means, stdev = self._norm(x)
        N = x_enc.shape[1]
        x_enc = x_enc.permute(0, 2, 1)
        x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels = self._encode(x_enc)
        x_dec = self._decode(x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denorm(dec_out, means, stdev)

        if self.clusters is None or len(self.clusters) != moving_mean.shape[-1]:
            self.clusters = self.calc_meanvar_clusters(moving_mean, self.n_clusters)
        moving_mean_out = self.trend_extractor(moving_mean, self.clusters)
        dec_out = dec_out + moving_mean_out
        return dec_out

    def super_resolution(self, x_enc):
        x, moving_mean = self.series_decomp(x_enc.permute(0, 2, 1))
        moving_mean = moving_mean.permute(0, 2, 1)
        x = self.eca_mrfp_block(x)
        x_enc, means, stdev = self._norm(x)
        N = x_enc.shape[1]
        x_enc = x_enc.permute(0, 2, 1)
        x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels = self._encode(x_enc)
        x_dec = self._decode(x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denorm(dec_out, means, stdev)

        if self.clusters is None or len(self.clusters) != moving_mean.shape[-1]:
            self.clusters = self.calc_meanvar_clusters(moving_mean, self.n_clusters)
        moving_mean_out = self.trend_extractor(moving_mean, self.clusters)
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
