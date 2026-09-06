# Edge Tensor Quantizer

A low-level, cache-aware tensor quantization and numerical linear algebra micro-kernel suite designed for resource-constrained embedded microcontrollers, edge NPUs, and CPU vector units.

## Architecture & Overview

```
[ FP32 Weights & Activations ] ──► [ Scale & Zero-Point Calibration ] ──► [ Quantized INT8 Buffer ]
                                      (MinMax / KL Divergence)                    │
                                                                                  ▼
                                                                     [ Tiled Cache-Aware GEMM ]
                                                                        (AVX2 / SIMD Vectorized)
```

## Features
- **Symmetric & Asymmetric INT8 Quantization**: Per-tensor and per-channel affine transformation mappings.
- **Hardware Cache Tiling**: Loop-tiled matrix multiplication (`GEMM`) designed to fit L1/L2 cache lines and eliminate TLB thrashing.
- **Dynamic Scale Calibration**: MinMax and histogram-based clipping thresholds to minimize quantization SNR loss.
- **Numerical Verification**: Direct Mean Squared Error (MSE) and Cosine Similarity evaluation harnesses.

## Tech Stack
- **Languages**: Python 3.11+ / C++ Interop
- **Numerics & Vectorization**: NumPy, SciPy
- **Testing & Benchmarks**: Pytest, Ruff

## Quickstart

```bash
# Install package
pip install -e .

# Run test suite
pytest tests/ -v
```
