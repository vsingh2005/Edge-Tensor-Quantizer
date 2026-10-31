import numpy as np
from edge_quantizer.residual import quantized_residual_add

def test_residual_add():
    a = np.array([50, -50], dtype=np.int8)
    b = np.array([25, -25], dtype=np.int8)
    res = quantized_residual_add(a, 0.1, b, 0.2, target_scale=0.1)
    assert res.dtype == np.int8
