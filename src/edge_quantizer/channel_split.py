"""
Channel splitting decomposition: split outlier channel into 2 channels with half magnitude.
"""
import numpy as np
from typing import Tuple

class ChannelSplitter:
    @staticmethod
    def split_channel(weights: np.ndarray, channel_idx: int) -> np.ndarray:
        # weights: (In_Channels, Out_Channels)
        in_c, out_c = weights.shape
        new_weights = np.empty((in_c + 1, out_c), dtype=weights.dtype)
        new_weights[:channel_idx] = weights[:channel_idx]
        new_weights[channel_idx] = weights[channel_idx] * 0.5
        new_weights[channel_idx + 1] = weights[channel_idx] * 0.5
        new_weights[channel_idx + 2:] = weights[channel_idx + 1:]
        return new_weights
