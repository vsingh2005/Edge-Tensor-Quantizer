import numpy as np
from edge_quantizer.int4 import Int4BlockQuantizer

def test_int4_pack_unpack_roundtrip():
    data = np.array([-3.5, -1.2, 0.0, 1.4, 3.2, 0.8, -2.1, 2.9], dtype=np.float32)
    quantizer = Int4BlockQuantizer(block_size=8)
    packed, scales = quantizer.quantize(data)
    assert packed.dtype == np.uint8
    assert packed.shape == (1, 4)
    reconstructed = quantizer.dequantize(packed, scales, original_length=len(data))
    assert np.allclose(data, reconstructed, atol=0.6)
