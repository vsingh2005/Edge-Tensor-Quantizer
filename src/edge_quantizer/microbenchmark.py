"""
Micro-benchmarking suite measuring GFLOPS and compression ratios.
"""
from typing import Dict, Any

class MicroBenchmark:
    @staticmethod
    def calculate_gflops(m: int, k: int, n: int, elapsed_seconds: float) -> float:
        total_ops = 2.0 * m * k * n
        if elapsed_seconds <= 0:
            return 0.0
        gflops = (total_ops / 1e9) / elapsed_seconds
        return round(gflops, 3)

    @staticmethod
    def calculate_compression_ratio(fp32_bytes: int, quant_bytes: int) -> Dict[str, Any]:
        ratio = (fp32_bytes / quant_bytes) if quant_bytes > 0 else 1.0
        savings_pct = (1.0 - (quant_bytes / fp32_bytes)) * 100.0 if fp32_bytes > 0 else 0.0
        return {
            "compression_ratio": round(ratio, 2),
            "memory_reduction_pct": round(savings_pct, 1)
        }
