# Edge-Tensor-Quantizer

A lightweight tensor quantization engine and low-level kernel benchmark suite exploring INT8, INT4, sub-byte representations, and cache-aware General Matrix Multiplication (GEMM) for resource-constrained edge hardware.

## Overview

Deep neural network deployment on edge processors and embedded microcontrollers is heavily constrained by SRAM capacity, memory bus bandwidth, and thermal envelopes. Standard 32-bit floating-point tensors consume excessive memory and require power-hungry floating-point units (FPUs).

This repository implements uniform and non-uniform tensor quantization algorithms, sub-byte nibble packing, layer sensitivity profiling, and cache-tiled matrix multiplication designed to minimize cache thrashing on modern CPU and NPU architectures.

## Architecture

```
+-------------------------------------------------------+
| Unquantized FP32 Tensor (Weights / Activations)       |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
| Quantization & Scaling Schemes                        |
|  - Symmetric INT8 (Scale factor, zero-point = 0)     |
|  - Asymmetric UINT8 (Scale + zero-point offset)       |
|  - Sub-Byte INT4 (Nibble bit-packing into uint8)      |
|  - NormalFloat4 (NF4) Quantile Quantization           |
|  - Microscaling FP4 (E2M1 Format Emulation)           |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
| Calibration & Precision Profiling                     |
|  - Per-Channel MSE & KL-Divergence Calibration        |
|  - Sensitivity-Based Mixed-Precision Layer Mapping    |
|  - Signal Quality Metrics: SNR (dB), Cosine Similarity|
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
| Hardware Execution Kernel                             |
|  - Cache-Aware Loop-Tiled GEMM (L1/L2 Cache Locality) |
|  - Benchmarked Throughput, GFLOPS, and Bandwidth      |
+-------------------------------------------------------+
```

## Core Modules and Engineering Details

- **Uniform Quantization (`quantize.py`, `asymmetric.py`)**: Implements symmetric signed INT8 quantization with dynamic scaling, as well as asymmetric unsigned UINT8 mapping with calculated zero-point offsets to handle skewed activation distributions.
- **Sub-Byte & Non-Uniform Representations (`int4.py`, `nf4.py`, `fp4.py`)**:
  - `int4.py`: Implements block-wise INT4 weight quantization with 4-bit nibble packing into `uint8` containers, halving the memory footprint compared to standard INT8.
  - `nf4.py`: Implements NormalFloat4 (NF4) quantile quantization for normally distributed weights, minimizing information loss in neural network weight tensors.
  - `fp4.py`: Emulates microscopic 4-bit floating-point formats (E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit).
- **Cache-Tiled GEMM Kernel (`gemm.py`)**: Implements loop-tiled General Matrix Multiplication with configurable sub-block sizes. Keeps active working tiles within L1/L2 CPU cache lines (64 bytes), avoiding memory bus bottlenecks and cache conflict misses during dense matrix operations.
- **Calibration & Profiling (`calibration.py`, `mixed_precision.py`)**: Includes histogram-based KL-divergence and Mean Squared Error (MSE) clipping threshold calibration. Provides mixed-precision sensitivity profiling to assign sensitive layers to higher bitwidths (INT8/FP16) while compressing resilient layers to INT4.
- **Accuracy & Degradation Metrics**: Measures tensor fidelity using Signal-to-Noise Ratio (SNR in dB), Cosine Similarity, and Peak Signal-to-Noise Ratio (PSNR) to guarantee mathematical correctness before deployment.

## Technical Decisions

- **Why Loop Tiling over Naive Triple-Loop MatMul**: Naive matrix multiplication exhibits poor cache locality on large matrices, resulting in frequent L1/L2 cache misses as row and column pointers traverse non-contiguous memory blocks. Tiling partitions matrices into sub-blocks that fit entirely within local CPU cache lines.
- **Why Per-Channel Scaling for Convolution and Linear Weights**: Per-tensor scaling maps all channels to a single scale factor, which leads to severe quantization error if one filter has a significantly larger dynamic range than others. Per-channel scaling computes independent scale factors per output slice, preserving precision across dynamic variance.
- **Why Asymmetric Zero-Point for Activations**: Neural network activations following ReLU or GELU functions are strictly non-negative or heavily skewed. Asymmetric quantization maps the active range precisely to [0, 255], doubling the effective representation density compared to symmetric signed ranges.

## Technology Stack

- **Language**: Python 3.11+, C++
- **Numerical Computation**: NumPy, SciPy
- **Benchmarking**: Custom timing and micro-profiling harnesses
- **Testing**: Pytest

## Project Structure

```
edge-tensor-quantizer/
├── src/
│   └── edge_quantizer/
│       ├── quantize.py        # Symmetric INT8 quantization
│       ├── asymmetric.py      # Asymmetric UINT8 with zero-point offset
│       ├── int4.py            # Sub-byte INT4 block-wise nibble packing
│       ├── nf4.py             # NormalFloat4 quantile quantization
│       ├── fp4.py             # Microscaling FP4 format emulation
│       ├── gemm.py            # Cache-tiled matrix multiplication
│       ├── calibration.py     # MSE and KL-divergence calibration
│       ├── mixed_precision.py # Sensitivity-based layer assignment
│       ├── benchmark.py       # Latency and memory bandwidth profiler
│       └── export.py          # Quantized tensor serialization
├── tests/                     # Precision and roundtrip numerical tests
├── pyproject.toml             # Build configuration
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git

### Installation

```bash
git clone https://github.com/vsingh2005/Edge-Tensor-Quantizer.git
cd Edge-Tensor-Quantizer
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -e .
```

### Running Tests

```bash
pytest tests/ -v
```