"""
Vectorized 8-way INT8 dot product accumulation avoiding overflow.
"""
import numpy as np

def simd_int8_dot_product(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape or a.dtype != np.int8 or b.dtype != np.int8:
        raise ValueError("Arrays must be equal shape int8")
    a_32 = a.astype(np.int32)
    b_32 = b.astype(np.int32)
    return int(np.dot(a_32, b_32))
