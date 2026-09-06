# Edge-Tensor-Quantizer

[![CI](https://github.com/vsingh2005/Edge-Tensor-Quantizer/actions/workflows/ci.yml/badge.svg)](https://github.com/vsingh2005/Edge-Tensor-Quantizer/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

Low-level tensor quantization and cache-aware linear algebra micro-kernels for edge NPUs and CPU vector units.

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
- **Numerical Verification**: Mean Squared Error (MSE) and Cosine Similarity evaluation harnesses.

## Tech Stack

Python 3.11+, C++ Interop, NumPy, SciPy, Pytest
