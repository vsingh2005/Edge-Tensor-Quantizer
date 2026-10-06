"""
Mixed-precision bitwidth allocator based on layer Hessian trace sensitivity.
"""
from typing import List, Dict, Any

class MixedPrecisionAssigner:
    @staticmethod
    def assign_bitwidths(
        layers: List[Dict[str, Any]],
        memory_budget_bytes: int
    ) -> List[Dict[str, Any]]:
        # Sort layers by sensitivity (highest sensitivity gets higher bitwidth)
        sorted_layers = sorted(layers, key=lambda layer: layer["hessian_trace"], reverse=True)
        # Default all to 4-bit, upgrade most sensitive to 8-bit or 16-bit
        for layer in sorted_layers:
            layer["bitwidth"] = 4

        def calc_size(layer_list):
            return sum(int(layer["num_params"] * (layer["bitwidth"] / 8.0)) for layer in layer_list)

        for layer in sorted_layers:
            layer["bitwidth"] = 8
            if calc_size(sorted_layers) > memory_budget_bytes:
                layer["bitwidth"] = 4
                break

        return sorted_layers
