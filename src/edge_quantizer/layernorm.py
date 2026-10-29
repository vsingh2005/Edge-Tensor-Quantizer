"""
Fused Layer Normalization + INT8 affine quantizer.
"""
import numpy as np
from typing import Tuple

def fused_layernorm_int8(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5
) -> Tuple[np.ndarray, float]:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + eps)
    scaled = normed * gamma + beta
    
    max_val = np.max(np.abs(scaled))
    scale = max(float(max_val) / 127.0, 1e-6)
    q = np.clip(np.round(scaled / scale), -128, 127).astype(np.int8)
    return q, scale
