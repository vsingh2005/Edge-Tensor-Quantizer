import numpy as np
from edge_quantizer.benchmark import QuantizationProfiler

def test_quantization_profiler():
    t = np.random.randn(256, 256).astype(np.float32)
    stats = QuantizationProfiler.profile_tensor_quantization(t, iterations=10)
    assert stats["compression_ratio"] >= 3.9
    assert stats["memory_reduction_pct"] > 70.0
    assert stats["latency_us"] > 0
