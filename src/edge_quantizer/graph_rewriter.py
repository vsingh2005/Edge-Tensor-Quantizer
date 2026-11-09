"""
Computational graph descriptor rewriter injecting QDQ nodes.
"""
from typing import List, Dict, Any

class GraphRewriter:
    @staticmethod
    def inject_qdq_nodes(graph_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rewritten = []
        for node in graph_nodes:
            op_type = node.get("op_type")
            if op_type in ["Conv", "MatMul", "Gemm"]:
                # Prepend QuantizeLinear and DequantizeLinear
                name = node.get("name", "node")
                rewritten.append({"name": f"{name}_quant", "op_type": "QuantizeLinear", "inputs": node["inputs"]})
                rewritten.append({"name": f"{name}_dequant", "op_type": "DequantizeLinear", "inputs": [f"{name}_quant"]})
                node_copy = dict(node)
                node_copy["inputs"] = [f"{name}_dequant"]
                rewritten.append(node_copy)
            else:
                rewritten.append(node)
        return rewritten
