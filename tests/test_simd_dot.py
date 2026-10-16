import numpy as np
from edge_quantizer.simd_dot import simd_int8_dot_product

def test_simd_dot():
    a = np.array([127, -128, 50, -50], dtype=np.int8)
    b = np.array([2, 2, 10, -10], dtype=np.int8)
    res = simd_int8_dot_product(a, b)
    assert res == (127 * 2 + -128 * 2 + 50 * 10 + -50 * -10)
