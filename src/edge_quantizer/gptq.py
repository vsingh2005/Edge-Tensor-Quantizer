"""
Inverse Hessian error compensation for one-shot weight quantization (GPTQ concept).
"""
import numpy as np

class GPTQCompensationSolver:
    @staticmethod
    def quantize_row(w: np.ndarray, H_inv: np.ndarray, scale: float) -> np.ndarray:
        q_w = np.zeros_like(w)
        w_cur = w.copy()
        for i in range(len(w)):
            q = np.clip(np.round(w_cur[i] / scale), -128, 127) * scale
            q_w[i] = q
            err = w_cur[i] - q
            h_diag = H_inv[i, i]
            if h_diag > 1e-6:
                # Disperse error to remaining columns
                w_cur[i:] -= (err / h_diag) * H_inv[i, i:]
        return q_w
