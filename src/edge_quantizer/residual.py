"""
Quantized residual addition with scale alignment and saturation clipping.
"""
import numpy as np
from typing import Tuple

def quantized_residual_add(
    A: np.ndarray, scale_a: float,
    B: np.ndarray, scale_b: float,
    target_scale: float
) -> np.ndarray:
    # A, B: int8
    real_a = A.astype(np.float32) * scale_a
    real_b = B.astype(np.float32) * scale_b
    summed = real_a + real_b
    q_out = np.clip(np.round(summed / target_scale), -128, 127).astype(np.int8)
    return q_out
