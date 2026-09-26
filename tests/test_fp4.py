import numpy as np
from edge_quantizer.fp4 import FP4Quantizer

def test_fp4_quantization_roundtrip():
    data = np.array([-5.8, -1.9, 0.1, 1.4, 2.9, 5.9], dtype=np.float32)
    indices, scale = FP4Quantizer.quantize(data)
    reconstructed = FP4Quantizer.dequantize(indices, scale)
    assert np.allclose(data, reconstructed, atol=0.6)
