from edge_quantizer.memory_profiler import MemoryFootprintProfiler

def test_memory_profiler():
    layers = [
        {"name": "conv1", "input_bytes": 10000, "weight_bytes": 5000, "output_bytes": 20000},
        {"name": "conv2", "input_bytes": 20000, "weight_bytes": 50000, "output_bytes": 20000},
    ]
    res = MemoryFootprintProfiler.profile_layers(layers)
    assert res["peak_bytes"] == 90000
    assert res["fits_in_512kb_sram"] is True
