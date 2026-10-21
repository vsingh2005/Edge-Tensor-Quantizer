import numpy as np
from edge_quantizer.kl_calibration import KLHistogramCalibrator

def test_kl_calibration():
    rng = np.random.RandomState(42)
    data = rng.randn(1000).astype(np.float32)
    threshold = KLHistogramCalibrator.calibrate_threshold(data)
    assert 0 < threshold <= np.max(np.abs(data))
