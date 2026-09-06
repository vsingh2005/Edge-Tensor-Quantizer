import numpy as np

class SymmetricQuantizer:
    @staticmethod
    def quantize(tensor: np.ndarray) -> tuple[np.ndarray, float]:
        max_val = float(np.max(np.abs(tensor)))
        scale = 1.0 if max_val == 0.0 else max_val / 127.0
        scaled = np.round(tensor / scale)
        q_tensor = np.clip(scaled, -128, 127).astype(np.int8)
        return q_tensor, scale

    @staticmethod
    def dequantize(q_tensor: np.ndarray, scale: float) -> np.ndarray:
        return q_tensor.astype(np.float32) * scale
