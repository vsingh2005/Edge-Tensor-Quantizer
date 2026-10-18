"""
Cache-friendly tiled GEMM with transposed B matrix to maximize L1 cache line hits.
"""
import numpy as np

def transposed_tiled_matmul(A: np.ndarray, B_t: np.ndarray, tile_size: int = 32) -> np.ndarray:
    M, K = A.shape
    N, K2 = B_t.shape
    if K != K2:
        raise ValueError("Inner dimensions must match")
    C = np.zeros((M, N), dtype=A.dtype)
    for i in range(0, M, tile_size):
        i_end = min(i + tile_size, M)
        for j in range(0, N, tile_size):
            j_end = min(j + tile_size, N)
            C[i:i_end, j:j_end] = np.dot(A[i:i_end, :], B_t[j:j_end, :].T)
    return C
