"""
Winograd F(2x2, 3x3) minimal filtering transform.
"""
import numpy as np

B_T = np.array([
    [1.0,  0.0, -1.0,  0.0],
    [0.0,  1.0,  1.0,  0.0],
    [0.0, -1.0,  1.0,  0.0],
    [0.0,  1.0,  0.0, -1.0]
], dtype=np.float32)

G = np.array([
    [ 1.0,  0.0,  0.0],
    [ 0.5,  0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.0,  0.0,  1.0]
], dtype=np.float32)

A_T = np.array([
    [1.0, 1.0,  1.0,  0.0],
    [0.0, 1.0, -1.0, -1.0]
], dtype=np.float32)

def winograd_f2x3_1d(d: np.ndarray, g: np.ndarray) -> np.ndarray:
    # d: (4,), g: (3,)
    V = B_T @ d
    U = G @ g
    M = U * V
    return A_T @ M
