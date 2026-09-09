import numpy as np

class AsymmetricQuantizer:
    @staticmethod
    def quantize(tensor: np.ndarray) -> tuple[np.ndarray, float, int]:
        min_val = float(np.min(tensor))
        max_val = float(np.max(tensor))
        if max_val == min_val:
            return np.zeros_like(tensor, dtype=np.uint8), 1.0, 0
        scale = (max_val - min_val) / 255.0
        zero_point = int(np.round(-min_val / scale))
        zero_point = max(0, min(255, zero_point))
        q_tensor = np.clip(np.round(tensor / scale) + zero_point, 0, 255).astype(np.uint8)
        return q_tensor, scale, zero_point

    @staticmethod
    def dequantize(q_tensor: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
        return (q_tensor.astype(np.float32) - zero_point) * scale
