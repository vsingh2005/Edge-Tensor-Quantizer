import numpy as np
from edge_quantizer.transposed_gemm import transposed_tiled_matmul

def test_transposed_gemm():
    A = np.random.randn(32, 16).astype(np.float32)
    B = np.random.randn(16, 24).astype(np.float32)
    B_t = B.T
    expected = A @ B
    res = transposed_tiled_matmul(A, B_t, tile_size=8)
    assert np.allclose(expected, res, atol=1e-5)
