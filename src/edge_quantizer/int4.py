import numpy as np

class Int4BlockQuantizer:
    def __init__(self, block_size: int = 32):
        if block_size <= 0 or block_size % 2 != 0:
            raise ValueError("block_size must be a positive even integer")
        self.block_size = block_size

    def quantize(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat = tensor.flatten().astype(np.float32)
        n = len(flat)
        pad_len = (self.block_size - (n % self.block_size)) % self.block_size
        if pad_len > 0:
            flat = np.pad(flat, (0, pad_len))
        num_blocks = len(flat) // self.block_size
        blocks = flat.reshape(num_blocks, self.block_size)
        scales = np.max(np.abs(blocks), axis=1) / 7.0
        scales[scales == 0.0] = 1.0

        q_blocks = np.clip(np.round(blocks / scales[:, None]), -8, 7).astype(np.int8)
        unsigned_q = (q_blocks + 8).astype(np.uint8)
        low = unsigned_q[:, 0::2]
        high = unsigned_q[:, 1::2]
        packed = ((high << 4) | low).astype(np.uint8)
        return packed, scales

    def dequantize(self, packed: np.ndarray, scales: np.ndarray, original_length: int) -> np.ndarray:
        low = (packed & 0x0F).astype(np.int8) - 8
        high = ((packed >> 4) & 0x0F).astype(np.int8) - 8
        unpacked = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.int8)
        unpacked[:, 0::2] = low
        unpacked[:, 1::2] = high
        reconstructed = (unpacked.astype(np.float32) * scales[:, None]).flatten()
        return reconstructed[:original_length]
