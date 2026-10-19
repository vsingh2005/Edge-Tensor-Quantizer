import numpy as np
from edge_quantizer.depthwise_conv import depthwise_conv2d_int8

def test_depthwise_conv2d():
    img = np.ones((2, 5, 5), dtype=np.int8)
    kernel = np.ones((2, 3, 3), dtype=np.int8)
    scales = np.array([0.1, 0.2], dtype=np.float32)
    out = depthwise_conv2d_int8(img, kernel, scales)
    assert out.shape == (2, 5, 5)
    # Center pixel sees 9 ones
    assert np.isclose(out[0, 2, 2], 0.9)
