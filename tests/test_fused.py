import numpy as np
from edge_quantizer.fused import FusedDenseKernel

def test_fused_dense_kernel_relu():
    x = np.array([[10, -20], [30, 40]], dtype=np.int8)
    w = np.array([[2, 1], [-1, 2]], dtype=np.int8)
    bias = np.array([0.0, 0.0], dtype=np.float32)
    out = FusedDenseKernel.forward(x, w, bias, x_scale=0.1, w_scale=0.1, out_scale=0.01, apply_relu=True)
    assert out.dtype == np.int8
    assert np.all(out >= 0)
