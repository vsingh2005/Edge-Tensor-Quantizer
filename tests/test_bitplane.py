import numpy as np
from edge_quantizer.bitplane import BitPlanePacker2Bit

def test_2bit_packing_roundtrip():
    data = np.array([0, 1, 2, 3, 3, 2, 1, 0, 2], dtype=np.uint8)
    packed = BitPlanePacker2Bit.pack(data)
    assert len(packed) == 3
    unpacked = BitPlanePacker2Bit.unpack(packed, original_length=len(data))
    assert np.array_equal(data, unpacked)
