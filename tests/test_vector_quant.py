import numpy as np
from edge_quantizer.vector_quant import VectorQuantizer

def test_vector_quantizer():
    data = np.linspace(-2.0, 2.0, 100).astype(np.float32)
    vq = VectorQuantizer(num_centroids=8)
    vq.fit(data)
    encoded = vq.encode(data)
    assert encoded.max() < 8
    decoded = vq.decode(encoded)
    assert np.allclose(data, decoded, atol=0.4)
