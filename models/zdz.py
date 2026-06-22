import torch
import numpy as np
import torch.nn as nn
from layers.Invertible import RevIN
from .yecamrdu import Model as YECAMRDUModel, ECABlock1D_MRFP


def _auto_kernel_size(enc_in: int) -> int:
    """
    为 ECA 自适应选择卷积核大小，随通道数增加但限制在较小范围。
    """
    if enc_in <= 16:
        return 3
    if enc_in <= 64:
        return 5
    return 7


def _auto_mrfp_ratio(enc_in: int, d_model: int, base_ratio: float) -> float:
    """
    根据输入通道和 d_model 自适应调整 MRFP 的隐藏宽度比例，控制在 [1, 4] 内。
    """
    # 以 d_model/enc_in 作为参考，取更大的那个，但限制范围
    ref = d_model / float(enc_in) if enc_in > 0 else base_ratio
    ratio = max(base_ratio, ref)
    return max(1.0, min(4.0, ratio))


class Model(YECAMRDUModel):
    """
    ZDZ: 在 YECAMRDU 基础上，让 ECA/MRFP 自适应并将趋势层替换为“均值/方差自适应聚类 + 线性”。
    """
    def __init__(self, configs):
        super().__init__(configs)
        channel = configs.enc_in
        self.n_clusters = configs.n_clusters
        base_ratio = getattr(configs, 'mrfp_ratio', 2.0)
        auto_ratio = _auto_mrfp_ratio(channel, getattr(configs, 'd_model', 512), base_ratio)
        auto_kernel = _auto_kernel_size(channel) if getattr(configs, 'eca_kernel_size', None) is None else configs.eca_kernel_size
        dropout = getattr(configs, 'dropout', 0.1)

        # 去除 ECA 注意力机制，改为恒等映射（保持接口一致）
        self.eca_mrfp_block = nn.Identity()

        # 自适应聚类的趋势层（替换 DUET 版本）
        self.trend_extractor = TrendClusteredLinearAdaptive(
            n_clusters=configs.n_clusters,
            seq_len=self.seq_len,
            pred_len=configs.pred_len,
            enc_in=configs.enc_in,
            dropout=dropout
        )
        self.clusters = None

    # ============ 聚类与趋势 ============ #
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

    # ============ 编码/解码辅助 ============ #
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

    # ============ 任务分支 ============ #
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
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


class TrendClusteredLinearAdaptive(nn.Module):
    """自适应通道聚类的趋势线性层（均值+方差分桶，每簇独立线性头）。"""
    def __init__(self, n_clusters, seq_len, pred_len, enc_in, dropout=0.1):
        super().__init__()
        self.n_clusters = n_clusters
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = RevIN(enc_in)
        self.linears = nn.ModuleDict({
            str(cid): nn.Sequential(
                nn.Linear(seq_len, seq_len * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(seq_len * 2, pred_len)
            ) for cid in range(n_clusters)
        })

    def forward(self, x, clusters):
        # x: (B, L, C); clusters: (C,)
        B, L, C = x.shape
        assert clusters.shape[0] == C, "clusters length must match channel count"
        x = self.revin(x, 'norm')
        outs = []
        for ch in range(C):
            cid = str(int(clusters[ch].item()))
            ch_data = x[:, :, ch]
            ch_out = self.linears[cid](ch_data)
            outs.append(ch_out.unsqueeze(-1))
        out = torch.cat(outs, dim=-1)  # (B, pred_len, C)
        out = self.revin(out, 'denorm')
        return out
