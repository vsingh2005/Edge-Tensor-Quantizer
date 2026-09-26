"""
Micro-scaling FP4 (E2M1) floating point quantizer.
1 sign bit, 2 exponent bits, 1 mantissa bit.
Representable values (positive): [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
"""
import numpy as np
from typing import Tuple

FP4_E2M1_LEVELS = np.array([
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0, 6.0
], dtype=np.float32)

class FP4Quantizer:
    @staticmethod
    def quantize(tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        max_abs = float(np.max(np.abs(tensor)))
        if max_abs == 0:
            return np.zeros_like(tensor, dtype=np.int8), 1.0
        scale = max_abs / 6.0
        normalized = tensor / scale
        
        # Nearest neighbor quantization to FP4 levels
        diffs = np.abs(normalized[..., None] - FP4_E2M1_LEVELS)
        indices = np.argmin(diffs, axis=-1).astype(np.int8)
        return indices, scale

    @staticmethod
    def dequantize(indices: np.ndarray, scale: float) -> np.ndarray:
        return FP4_E2M1_LEVELS[indices] * scale
