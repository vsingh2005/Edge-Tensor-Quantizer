"""
NormalFloat4 (NF4) optimal quantile-spaced 4-bit quantizer for normally-distributed weights.
"""
import numpy as np
from typing import Tuple

NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
     0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
     0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=np.float32)

class NF4Quantizer:
    @staticmethod
    def quantize(weights: np.ndarray) -> Tuple[np.ndarray, float]:
        norm_factor = float(np.max(np.abs(weights)))
        if norm_factor == 0:
            return np.zeros_like(weights, dtype=np.uint8), 1.0
        normalized = weights / norm_factor
        diffs = np.abs(normalized[..., None] - NF4_LEVELS)
        indices = np.argmin(diffs, axis=-1).astype(np.uint8)
        return indices, norm_factor

    @staticmethod
    def dequantize(indices: np.ndarray, scale: float) -> np.ndarray:
        return NF4_LEVELS[indices] * scale
