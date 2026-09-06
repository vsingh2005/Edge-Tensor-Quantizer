"""
Unit tests for symmetric quantization.
"""
import numpy as np
from edge_quantizer.quantize import SymmetricQuantizer

def test_symmetric_quantization_roundtrip():
    original = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    q_tensor, scale = SymmetricQuantizer.quantize(original)
    
    assert q_tensor.dtype == np.int8
    assert scale > 0.0
    
    reconstructed = SymmetricQuantizer.dequantize(q_tensor, scale)
    assert np.allclose(original, reconstructed, atol=0.02)
