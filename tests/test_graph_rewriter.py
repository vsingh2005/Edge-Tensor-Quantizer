from edge_quantizer.graph_rewriter import GraphRewriter

def test_qdq_injection():
    nodes = [{"name": "dense1", "op_type": "Gemm", "inputs": ["input0"]}]
    rewritten = GraphRewriter.inject_qdq_nodes(nodes)
    assert len(rewritten) == 3
    assert rewritten[0]["op_type"] == "QuantizeLinear"
