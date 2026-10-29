"""
Quantized Scaled Dot-Product Attention kernel.
"""
import numpy as np

def quantized_scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float
) -> np.ndarray:
    # Q, K, V: (Seq, D)
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    # Softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return (weights @ V) * scale
