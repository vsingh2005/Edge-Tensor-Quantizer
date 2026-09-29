import numpy as np
from edge_quantizer.nf4 import NF4Quantizer

def test_nf4_roundtrip():
    rng = np.random.RandomState(42)
    weights = rng.randn(100).astype(np.float32)
    idx, scale = NF4Quantizer.quantize(weights)
    assert idx.shape == weights.shape
    recon = NF4Quantizer.dequantize(idx, scale)
    assert np.allclose(weights, recon, atol=0.4)
