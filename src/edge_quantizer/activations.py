"""
Fast polynomial approximation of Gaussian Error Linear Unit (GELU) with fused bias.
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
"""
import numpy as np

SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))

def fused_bias_gelu(x: np.ndarray, bias: np.ndarray) -> np.ndarray:
    xb = x + bias
    inner = SQRT_2_OVER_PI * (xb + 0.044715 * (xb ** 3))
    return 0.5 * xb * (1.0 + np.tanh(inner))
