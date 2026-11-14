from edge_quantizer.microbenchmark import MicroBenchmark

def test_microbenchmark():
    gflops = MicroBenchmark.calculate_gflops(100, 100, 100, 0.001)
    assert gflops > 0
    res = MicroBenchmark.calculate_compression_ratio(fp32_bytes=4000, quant_bytes=1000)
    assert res["compression_ratio"] == 4.0
    assert res["memory_reduction_pct"] == 75.0
