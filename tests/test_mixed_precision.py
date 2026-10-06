from edge_quantizer.mixed_precision import MixedPrecisionAssigner

def test_mixed_precision():
    layers = [
        {"name": "conv1", "num_params": 1000, "hessian_trace": 150.0},
        {"name": "conv2", "num_params": 2000, "hessian_trace": 50.0},
    ]
    assigned = MixedPrecisionAssigner.assign_bitwidths(layers, memory_budget_bytes=1600)
    assert assigned[0]["bitwidth"] >= assigned[1]["bitwidth"]
