"""
Core tensor quantization algorithms (FP32 to INT8 symmetric/asymmetric mapping).
"""
import numpy as np
from typing import Tuple, Dict, Any

class SymmetricQuantizer:
    """
    Symmetric linear INT8 quantizer: Q(x) = clip(round(x / scale), -128, 127)
    where scale = max(abs(x)) / 127.0
    """
    @staticmethod
    def quantize(tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """Quantizes an FP32 ndarray to INT8 with computed scale."""
        max_val = float(np.max(np.abs(tensor)))
        if max_val == 0.0:
            scale = 1.0
        else:
            scale = max_val / 127.0
            
        scaled = np.round(tensor / scale)
        q_tensor = np.clip(scaled, -128, 127).astype(np.int8)
        return q_tensor, scale

    @staticmethod
    def dequantize(q_tensor: np.ndarray, scale: float) -> np.ndarray:
        """Dequantizes an INT8 ndarray back to reconstructed FP32."""
        return q_tensor.astype(np.float32) * scale
