import torch

from .yecamrdu import Model as BaseModel


class Model(BaseModel):
    """
    ZTONGFEN: 基于 yecamrdu 的列级预测版本。

    设计目的：削弱多通道之间的耦合影响。每次仅保留单列输入做一次预测，
    将各列预测结果拼接回完整输出；若提供目标序列，可计算逐列 MAE/MSE
    并在全部列结束后取平均值。
    """

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None,
        target=None,
        return_metrics: bool = False,
    ):
        # 非预测任务直接复用基类行为，保持兼容
        if self.task_name not in ("long_term_forecast", "short_term_forecast"):
            base_out = super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
            if return_metrics:
                return base_out, None, None
            return base_out

        num_channels = x_enc.shape[-1]
        channel_preds = []
        channel_mae = []
        channel_mse = []

        for ch in range(num_channels):
            # 仅保留当前列，其余列置零，弱化通道间交互
            ch_x_enc = torch.zeros_like(x_enc)
            ch_x_dec = torch.zeros_like(x_dec)
            ch_x_enc[:, :, ch] = x_enc[:, :, ch]
            ch_x_dec[:, :, ch] = x_dec[:, :, ch]

            # 基类 forward 已包含裁剪到 pred_len 的逻辑
            pred_full = super().forward(ch_x_enc, x_mark_enc, ch_x_dec, x_mark_dec, mask=mask)
            pred_col = pred_full[:, :, ch : ch + 1]
            channel_preds.append(pred_col)

            if target is not None:
                true_col = target[:, :, ch : ch + 1]
                channel_mae.append(torch.mean(torch.abs(pred_col - true_col)))
                channel_mse.append(torch.mean((pred_col - true_col) ** 2))

        merged_pred = torch.cat(channel_preds, dim=-1)

        avg_mae = avg_mse = None
        if target is not None and channel_mae:
            avg_mae = torch.stack(channel_mae).mean()
            avg_mse = torch.stack(channel_mse).mean()
            # 便于外部读取最近一次的列均值指标
            self.latest_channel_mae = avg_mae.detach().item()
            self.latest_channel_mse = avg_mse.detach().item()
        else:
            self.latest_channel_mae = None
            self.latest_channel_mse = None

        if return_metrics:
            return merged_pred, avg_mae, avg_mse
        return merged_pred
