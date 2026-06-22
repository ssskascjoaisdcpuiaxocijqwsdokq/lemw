import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.LiftingScheme import LiftingScheme, InverseLiftingScheme
from layers.Invertible import RevIN
from layers.Autoformer_EncDec import series_decomp as _autoformer_series_decomp
from einops import rearrange


def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series"""
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2))
        end = x[:, :, -1:].repeat(1, 1, math.floor((self.kernel_size - 1) // 2))
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x


class moving_avg_imputation(nn.Module):
    """Moving average block modified to ignore zeros in the moving window."""
    def __init__(self, kernel_size, stride):
        super(moving_avg_imputation, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        num_channels = x.shape[1]
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2))
        end = x[:, :, -1:].repeat(1, 1, math.floor((self.kernel_size - 1) // 2))
        x_padded = torch.cat([front, x, end], dim=-1)

        non_zero_mask = x_padded != 0
        weight = torch.ones((1, num_channels, self.kernel_size), device=x_padded.device)
        window_sum = F.conv1d(x_padded, weight=weight, stride=self.stride)
        window_count = F.conv1d(non_zero_mask.float(), weight=weight, stride=self.stride)
        window_count = torch.clamp(window_count, min=1)
        moving_avg = window_sum / window_count
        return moving_avg


class series_decomp(nn.Module):
    """Series decomposition block"""
    def __init__(self, kernel_size=24, stride=1, imputation=False):
        super(series_decomp, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.moving_avg = moving_avg(kernel_size, stride=stride) if not imputation else moving_avg_imputation(self.kernel_size, self.stride)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class AdpWaveletBlock(nn.Module):
    def __init__(self, configs, input_size):
        super(AdpWaveletBlock, self).__init__()
        self.regu_details = configs.regu_details
        self.regu_approx = configs.regu_approx
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingScheme(configs.enc_in, k_size=configs.lifting_kernel_size, input_size=input_size)
        self.norm_x = normalization(configs.enc_in)
        self.norm_d = normalization(configs.enc_in)

    def forward(self, x):
        c, d = self.wavelet(x)
        x = c

        r = None
        if self.regu_approx + self.regu_details != 0.0:
            rd = self.regu_details * d.abs().mean() if self.regu_details else 0.0
            rc = self.regu_approx * torch.dist(c.mean(), x.mean(), p=2) if self.regu_approx else 0.0
            r = rd + rc if self.regu_details and self.regu_approx else (rd or rc)

        x = self.norm_x(x)
        d = self.norm_d(d)
        return x, r, d


class InverseAdpWaveletBlock(nn.Module):
    def __init__(self, configs, input_size):
        super(InverseAdpWaveletBlock, self).__init__()
        self.inverse_wavelet = InverseLiftingScheme(configs.enc_in, input_size=input_size, kernel_size=configs.lifting_kernel_size)

    def forward(self, c, d):
        reconstructed = self.inverse_wavelet(c, d)
        return reconstructed


class DUETLinearExtractor(nn.Module):
    """
    基于 DUET 的趋势提取器：RevIN → temporal projection → MLP → pred_len，通道独立。
    """
    def __init__(self, seq_len, pred_len, enc_in, d_model=512, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model

        self.revin = RevIN(enc_in)
        self.temporal_projection = nn.Linear(seq_len, d_model)
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, pred_len),
            nn.Dropout(dropout)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        x_norm = self.revin(x, 'norm')
        x_reshaped = rearrange(x_norm, 'b l c -> (b c) l 1')
        temporal_features = self.temporal_projection(x_reshaped.squeeze(-1))
        extracted_features = self.feature_extractor(temporal_features)
        predictions = self.prediction_head(extracted_features)
        predictions = rearrange(predictions, '(b c) p -> b p c', b=B, c=C)
        predictions = self.revin(predictions, 'denorm')
        return predictions


class Model(nn.Module):
    """
    YEMRDU（独立版）：在 YECAMRDU 基础上去掉 ECA 注意力，使用恒等映射，其他结构保留。
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

        # 趋势提取：DUET 线性提取器（通道独立）
        self.trend_extractor = DUETLinearExtractor(
            seq_len=self.seq_len,
            pred_len=configs.pred_len,
            enc_in=configs.enc_in,
            d_model=getattr(configs, 'duet_d_model', 512),
            dropout=configs.dropout
        )

        # 去掉 ECA 注意力，使用恒等映射
        self.eca_mrfp_block = nn.Identity()

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len // (2 ** configs.lifting_levels), configs.d_model, configs.embed, configs.freq,
            configs.dropout)

        # 小波编码层级
        self.encoder_levels = nn.ModuleList()
        self.linear_levels = nn.ModuleList()
        self.coef_linear_levels = nn.ModuleList()
        self.coef_dec_levels = nn.ModuleList()
        input_size = self.seq_len
        expand_ratio = configs.sr_ratio if self.task_name == "super_resolution" else 1

        for i in range(configs.lifting_levels):
            self.encoder_levels.add_module(
                'encoder_level_' + str(i),
                AdpWaveletBlock(configs, input_size)
            )
            input_size = input_size // 2
            self.linear_levels.add_module(
                'linear_level_' + str(i),
                nn.Sequential(
                    nn.Linear(input_size, input_size * expand_ratio),
                )
            )
            self.coef_linear_levels.add_module(
                'linear_level_' + str(i),
                nn.Sequential(
                    nn.Linear(input_size, input_size * expand_ratio),
                )
            )
            self.coef_dec_levels.add_module(
                'linear_level_' + str(i),
                nn.Sequential(
                    nn.Linear(input_size, input_size * expand_ratio),
                )
            )

        self.input_size = input_size

        # 解码层级
        self.decoder_levels = nn.ModuleList()
        for i in range(configs.lifting_levels - 1, -1, -1):
            self.decoder_levels.add_module(
                'decoder_level_' + str(i),
                InverseAdpWaveletBlock(configs, input_size=input_size)
            )
            input_size *= 2

        if self.task_name == "super_resolution":
            self.lowrank_projection = nn.Linear(configs.d_model, self.pred_len // (2 ** configs.lifting_levels), bias=True)
        else:
            self.lowrank_projection = nn.Linear(configs.d_model, self.seq_len // (2 ** configs.lifting_levels), bias=True)

        # Encoder
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
        # Projection
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(self.seq_len, configs.pred_len, bias=True)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(self.seq_len, self.seq_len, bias=True)
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(self.seq_len, self.seq_len, bias=True)
        elif self.task_name == 'super_resolution':
            self.projection = nn.Linear(configs.pred_len, configs.pred_len, bias=True)

    # ============ 公共流程 ============ #
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

    # ============ 任务分支 ============ #
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x, moving_mean = self.series_decomp(x_enc.permute(0, 2, 1))
        moving_mean = moving_mean.permute(0, 2, 1)
        x = self.eca_mrfp_block(x)

        x_enc, means, stdev = self._norm(x)
        _, _, N = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)
        x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels = self._encode(x_enc)
        x_dec = self._decode(x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
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
        x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels = self._encode(x_enc)
        x_dec = self._decode(x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
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
        x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels = self._encode(x_enc)
        x_dec = self._decode(x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
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
        x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels = self._encode(x_enc)
        x_dec = self._decode(x_enc, encoded_coefficients, x_embedding_levels, coef_embedding_levels)
        dec_out = self.projection(x_dec).permute(0, 2, 1)[:, :, :N]
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
