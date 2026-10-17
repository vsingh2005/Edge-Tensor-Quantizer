"""
Interleaved 2-bit weight packing into 8-bit unsigned integers (4 weights per byte).
"""
import numpy as np

class BitPlanePacker2Bit:
    @staticmethod
    def pack(weights: np.ndarray) -> np.ndarray:
        flat = weights.flatten()
        pad = (4 - (len(flat) % 4)) % 4
        if pad > 0:
            flat = np.pad(flat, (0, pad))
        flat = np.clip(flat, 0, 3).astype(np.uint8)
        w0 = flat[0::4]
        w1 = flat[1::4]
        w2 = flat[2::4]
        w3 = flat[3::4]
        return ((w3 << 6) | (w2 << 4) | (w1 << 2) | w0).astype(np.uint8)

    @staticmethod
    def unpack(packed: np.ndarray, original_length: int) -> np.ndarray:
        w0 = packed & 0x03
        w1 = (packed >> 2) & 0x03
        w2 = (packed >> 4) & 0x03
        w3 = (packed >> 6) & 0x03
        unpacked = np.empty((len(packed), 4), dtype=np.uint8)
        unpacked[:, 0] = w0
        unpacked[:, 1] = w1
        unpacked[:, 2] = w2
        unpacked[:, 3] = w3
        return unpacked.flatten()[:original_length]
