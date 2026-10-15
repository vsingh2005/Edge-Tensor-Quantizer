import numpy as np
from edge_quantizer.smoothquant import SmoothQuant

def test_smoothquant_equivalence():
    X = np.array([[10.0, 1.0], [5.0, 2.0]], dtype=np.float32)
    W = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scale = SmoothQuant.compute_smoothing_scale(np.array([10.0, 2.0]), np.array([3.0, 4.0]))
    X_s, W_s = SmoothQuant.apply_smoothing(X, W, scale)
    original_out = X @ W
    smooth_out = X_s @ W_s
    assert np.allclose(original_out, smooth_out, atol=1e-5)
