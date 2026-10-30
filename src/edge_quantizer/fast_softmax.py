"""
Integer softmax using a precomputed 256-entry lookup table.
"""
import numpy as np

EXP_LUT = np.exp(-np.linspace(0, 8, 256)).astype(np.float32)

class FastIntegerSoftmax:
    @staticmethod
    def softmax_lut(logits: np.ndarray) -> np.ndarray:
        max_val = np.max(logits, axis=-1, keepdims=True)
        diff = np.clip(max_val - logits, 0.0, 8.0)
        indices = np.clip(np.round(diff * (255.0 / 8.0)), 0, 255).astype(np.int32)
        exp_vals = EXP_LUT[indices]
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
