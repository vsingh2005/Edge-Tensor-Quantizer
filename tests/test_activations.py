import numpy as np
from edge_quantizer.activations import fused_bias_gelu

def test_fused_bias_gelu():
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    b = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    out = fused_bias_gelu(x, b)
    assert out[1] == 0.0
    assert out[2] > 0.8
