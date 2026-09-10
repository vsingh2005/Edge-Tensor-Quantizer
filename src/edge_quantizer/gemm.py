import numpy as np

def tiled_matmul(A: np.ndarray, B: np.ndarray, tile_size: int = 32) -> np.ndarray:
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Dimension mismatch: {A.shape} vs {B.shape}")
    C = np.zeros((M, N), dtype=A.dtype)
    for i0 in range(0, M, tile_size):
        i_max = min(i0 + tile_size, M)
        for j0 in range(0, N, tile_size):
            j_max = min(j0 + tile_size, N)
            for k0 in range(0, K, tile_size):
                k_max = min(k0 + tile_size, K)
                C[i0:i_max, j0:j_max] += np.dot(A[i0:i_max, k0:k_max], B[k0:k_max, j0:j_max])
    return C
