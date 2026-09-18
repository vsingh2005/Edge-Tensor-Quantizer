import time
from typing import Dict, Any
import numpy as np
from edge_quantizer.quantize import SymmetricQuantizer

class QuantizationProfiler:
    @staticmethod
    def profile_tensor_quantization(tensor: np.ndarray, iterations: int = 100) -> Dict[str, Any]:
        fp32_bytes = tensor.nbytes
        q, scale = SymmetricQuantizer.quantize(tensor)
        int8_bytes = q.nbytes + 4
        start_time = time.perf_counter()
        for _ in range(iterations):
            SymmetricQuantizer.quantize(tensor)
        elapsed_us = ((time.perf_counter() - start_time) / iterations) * 1_000_000
        compression_ratio = fp32_bytes / max(1, int8_bytes)
        return {
            "fp32_bytes": fp32_bytes,
            "int8_bytes": int8_bytes,
            "compression_ratio": round(compression_ratio, 2),
            "memory_reduction_pct": round((1.0 - (int8_bytes / fp32_bytes)) * 100, 2),
            "latency_us": round(elapsed_us, 2)
        }
