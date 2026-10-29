import numpy as np
from edge_quantizer.attention import quantized_scaled_dot_product_attention

def test_quantized_attention():
    Q = np.random.randn(4, 8).astype(np.float32)
    K = np.random.randn(4, 8).astype(np.float32)
    V = np.random.randn(4, 8).astype(np.float32)
    out = quantized_scaled_dot_product_attention(Q, K, V, scale=1.0)
    assert out.shape == (4, 8)
