import numpy as np
from edge_quantizer.calibration import CalibrationEngine

def test_optimal_threshold_mse():
    np.random.seed(42)
    tensor = np.random.randn(1000).astype(np.float32)
    tensor[0] = 50.0
    threshold = CalibrationEngine.find_optimal_threshold_mse(tensor)
    assert threshold < 50.0
    assert threshold > 1.0

def test_calibrate_per_channel():
    weights = np.random.randn(4, 64).astype(np.float32)
    scales = CalibrationEngine.calibrate_weights_per_channel(weights)
    assert len(scales) == 4
    assert np.all(scales > 0)
