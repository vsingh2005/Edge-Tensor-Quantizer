import numpy as np
from edge_quantizer.fast_softmax import FastIntegerSoftmax

def test_fast_softmax():
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    probs = FastIntegerSoftmax.softmax_lut(logits)
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[0, 2] > probs[0, 0]
