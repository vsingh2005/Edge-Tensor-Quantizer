import numpy as np
from edge_quantizer.winograd import winograd_f2x3_1d

def test_winograd_1d():
    d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    g = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    y = winograd_f2x3_1d(d, g)
    assert len(y) == 2
