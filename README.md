# Edge-Tensor-Quantizer

A lightweight tensor quantization and matrix multiplication experiment in Python and C++.

## What it does

- **INT8 Quantization**: Converts 32-bit floating point tensors to 8-bit integers using symmetric and asymmetric scale/zero-point mapping.
- **Per-Channel Scaling**: Calculates individual scale factors per output channel for convolutional or linear weights.
- **Cache-Tiled MatMul**: Implements loop-tiled matrix multiplication (GEMM) to reduce cache misses by keeping active sub-blocks in L1/L2 cache.
- **Accuracy Metrics**: Measures signal degradation using Mean Squared Error (MSE), SNR (dB), and Cosine Similarity.

## Stack

Python, C++, NumPy, SciPy