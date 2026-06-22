from .yecamrdu import Model as BaseModel


class Model(BaseModel):
    """
    ZYECAMRDU: 在 yecamrdu 基础上，仅在保存（state_dict 导出）时打印 ECA+MRFP gate 数值。
    """
    def state_dict(self, *args, **kwargs):
        if not getattr(self, "_gate_printed_on_save", False):
            if hasattr(self, "eca_mrfp_block") and hasattr(self.eca_mrfp_block, "gate"):
                gate_tensor = self.eca_mrfp_block.gate
                try:
                    gate_val = gate_tensor.detach().cpu().item()
                except Exception:
                    gate_val = gate_tensor
                print(f"[ZYECAMRDU] gate (on save) = {gate_val}")
            else:
                print("[ZYECAMRDU] gate attribute not found in eca_mrfp_block (on save).")
            self._gate_printed_on_save = True
        return super().state_dict(*args, **kwargs)
