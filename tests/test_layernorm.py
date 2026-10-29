import numpy as np
from edge_quantizer.layernorm import fused_layernorm_int8

def test_fused_layernorm():
    x = np.random.randn(2, 8).astype(np.float32)
    gamma = np.ones(8, dtype=np.float32)
    beta = np.zeros(8, dtype=np.float32)
    q, scale = fused_layernorm_int8(x, gamma, beta)
    assert q.dtype == np.int8
    assert q.shape == x.shape
