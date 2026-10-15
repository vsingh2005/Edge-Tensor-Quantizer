"""
SmoothQuant: Mathematical activation-weight migration scaling.
Scale s = max(|X|)^alpha / max(|W|)^(1 - alpha)
X' = X * diag(s)^(-1)
W' = diag(s) * W
"""
import numpy as np
from typing import Tuple

class SmoothQuant:
    @staticmethod
    def compute_smoothing_scale(
        act_max: np.ndarray,
        weight_max: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        act_pow = np.power(np.maximum(act_max, 1e-5), alpha)
        weight_pow = np.power(np.maximum(weight_max, 1e-5), 1.0 - alpha)
        scale = act_pow / weight_pow
        scale[scale == 0] = 1.0
        return scale

    @staticmethod
    def apply_smoothing(
        X: np.ndarray,
        W: np.ndarray,
        scale: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # X: (M, K), W: (K, N), scale: (K,)
        X_smooth = X / scale[None, :]
        W_smooth = W * scale[:, None]
        return X_smooth, W_smooth
