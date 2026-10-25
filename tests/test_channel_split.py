import numpy as np
from edge_quantizer.channel_split import ChannelSplitter

def test_channel_splitting():
    W = np.array([[2.0], [10.0], [4.0]])
    W_split = ChannelSplitter.split_channel(W, channel_idx=1)
    assert W_split.shape == (4, 1)
    assert W_split[1, 0] == 5.0
    assert W_split[2, 0] == 5.0
