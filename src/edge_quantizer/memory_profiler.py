"""
Peak SRAM memory footprint and ping-pong double buffer allocator simulator.
"""
from typing import List, Dict, Any

class MemoryFootprintProfiler:
    @staticmethod
    def profile_layers(layers: List[Dict[str, int]]) -> Dict[str, Any]:
        peak_bytes = 0
        timeline = []
        for layer in layers:
            # Memory = input_bytes + weight_bytes + output_bytes
            mem = layer.get("input_bytes", 0) + layer.get("weight_bytes", 0) + layer.get("output_bytes", 0)
            if mem > peak_bytes:
                peak_bytes = mem
            timeline.append({"layer": layer.get("name"), "bytes": mem})
            
        return {
            "peak_bytes": peak_bytes,
            "peak_kb": round(peak_bytes / 1024.0, 2),
            "fits_in_512kb_sram": peak_bytes <= (512 * 1024)
        }
