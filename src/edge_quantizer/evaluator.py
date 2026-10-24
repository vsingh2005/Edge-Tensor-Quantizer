"""
Quantization error metrics: SQNR and Cosine Similarity.
"""
import numpy as np

class QuantizationEvaluator:
    @staticmethod
    def compute_sqnr(original: np.ndarray, quantized: np.ndarray) -> float:
        signal_power = np.mean(original.astype(np.float64) ** 2)
        noise_power = np.mean((original.astype(np.float64) - quantized.astype(np.float64)) ** 2)
        if noise_power == 0:
            return 99.99
        return float(10.0 * np.log10(signal_power / noise_power))

    @staticmethod
    def compute_cosine_similarity(original: np.ndarray, quantized: np.ndarray) -> float:
        o_flat = original.flatten().astype(np.float64)
        q_flat = quantized.flatten().astype(np.float64)
        denom = np.linalg.norm(o_flat) * np.linalg.norm(q_flat)
        if denom == 0:
            return 1.0
        return float(np.dot(o_flat, q_flat) / denom)
