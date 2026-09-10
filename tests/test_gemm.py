import numpy as np
from edge_quantizer.gemm import tiled_matmul

def test_tiled_matmul_correctness():
    A = np.random.randn(64, 64).astype(np.float32)
    B = np.random.randn(64, 64).astype(np.float32)
    expected = np.dot(A, B)
    result = tiled_matmul(A, B, tile_size=16)
    assert np.allclose(expected, result, atol=1e-5)
