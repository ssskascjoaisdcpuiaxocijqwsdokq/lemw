import torch

from .yecamrdu import Model as BaseModel


class Model(BaseModel):
    """
    ZYECAMEDUXX:
    在 `yecamrdu` 基础上，将 Non-stationary Transformer 的归一化步骤与 ECA+MRFP 注意力块交换顺序：
    先归一化（mean/std），再做 `eca_mrfp_block`。
    """

    def _norm_then_eca_mrfp(self, x):
        """
        x: (B, C, L) seasonal component after decomposition.
        returns: x_enc (B, C, L) after normalization and attention, plus (means, stdev, N)
        """
        x_enc = x.permute(0, 2, 1)  # (B, L, C)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        _, _, N = x_enc.shape

        x_enc = x_enc.permute(0, 2, 1)  # (B, C, L)
        x_enc = self.eca_mrfp_block(x_enc)
        return x_enc, means, stdev, N

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x, moving_mean = self.series_decomp(x_enc.permute(0, 2, 1))
        moving_mean = moving_mean.permute(0, 2, 1)

        # Swap order: normalize first, then ECA+MRFP
        x_enc, means, stdev, N = self._norm_then_eca_mrfp(x)

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
        x_enc = x_enc.permute(0, 2, 1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        # Decoding
        for dec, x_emb_level, coef_emb_level, c_linear in zip(
            self.decoder_levels,
            x_embedding_levels[::-1],
            coef_embedding_levels[::-1],
            self.coef_dec_levels[::-1],
        ):
            details = encoded_coefficients.pop()
            details = coef_emb_level + c_linear(details)
            x_dec = x_dec + x_emb_level
            x_dec = dec(x_dec, details)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        moving_mean_out = self.trend_extractor(moving_mean)
        dec_out = dec_out + moving_mean_out
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x, moving_mean = self.series_decomp(x_enc.permute(0, 2, 1))
        moving_mean = moving_mean.permute(0, 2, 1)

        # Swap order: normalize first, then ECA+MRFP
        x_enc, means, stdev, N = self._norm_then_eca_mrfp(x)

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
        x_enc = x_enc.permute(0, 2, 1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        # Decoding
        for dec, x_emb_level, coef_emb_level, c_linear in zip(
            self.decoder_levels,
            x_embedding_levels[::-1],
            coef_embedding_levels[::-1],
            self.coef_dec_levels[::-1],
        ):
            details = encoded_coefficients.pop()
            details = coef_emb_level + c_linear(details)
            x_dec = x_dec + x_emb_level
            x_dec = dec(x_dec, details)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        moving_mean_out = self.trend_extractor(moving_mean)
        dec_out = dec_out + moving_mean_out
        return dec_out

    def anomaly_detection(self, x_enc):
        x, moving_mean = self.series_decomp(x_enc.permute(0, 2, 1))
        moving_mean = moving_mean.permute(0, 2, 1)

        # Swap order: normalize first, then ECA+MRFP
        x_enc, means, stdev, N = self._norm_then_eca_mrfp(x)

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
        x_enc = x_enc.permute(0, 2, 1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        # Decoding
        for dec, x_emb_level, coef_emb_level, c_linear in zip(
            self.decoder_levels,
            x_embedding_levels[::-1],
            coef_embedding_levels[::-1],
            self.coef_dec_levels[::-1],
        ):
            details = encoded_coefficients.pop()
            details = coef_emb_level + c_linear(details)
            x_dec = x_dec + x_emb_level
            x_dec = dec(x_dec, details)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        moving_mean_out = self.trend_extractor(moving_mean)
        dec_out = dec_out + moving_mean_out
        return dec_out

    def super_resolution(self, x_enc):
        x, moving_mean = self.series_decomp(x_enc.permute(0, 2, 1))
        moving_mean = moving_mean.permute(0, 2, 1)

        # Swap order: normalize first, then ECA+MRFP
        x_enc, means, stdev, N = self._norm_then_eca_mrfp(x)

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
        x_enc = x_enc.permute(0, 2, 1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        x_dec = self.lowrank_projection(enc_out)
        # Decoding
        for dec, x_emb_level, coef_emb_level, c_linear in zip(
            self.decoder_levels,
            x_embedding_levels[::-1],
            coef_embedding_levels[::-1],
            self.coef_dec_levels[::-1],
        ):
            details = encoded_coefficients.pop()
            details = coef_emb_level + c_linear(details)
            x_dec = x_dec + x_emb_level
            x_dec = dec(x_dec, details)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        moving_mean_out = self.trend_extractor(moving_mean)
        dec_out = dec_out + moving_mean_out
        return dec_out

