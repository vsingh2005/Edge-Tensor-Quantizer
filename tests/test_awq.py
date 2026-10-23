import numpy as np
from edge_quantizer.awq import AWQSalienceSearch

def test_awq_salience():
    acts = np.array([[1.0, 50.0, 2.0], [0.5, 45.0, 1.2]], dtype=np.float32)
    salient = AWQSalienceSearch.identify_salient_channels(acts, top_k_pct=0.34)
    assert 1 in salient
