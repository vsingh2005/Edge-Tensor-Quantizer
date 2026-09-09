import numpy as np
from edge_quantizer.asymmetric import AsymmetricQuantizer

def test_asymmetric_quantization_roundtrip():
    data = np.array([0.0, 1.5, 3.0, 4.5, 6.0], dtype=np.float32)
    q, scale, zp = AsymmetricQuantizer.quantize(data)
    assert q.dtype == np.uint8
    reconstructed = AsymmetricQuantizer.dequantize(q, scale, zp)
    assert np.allclose(data, reconstructed, atol=0.05)
