"""
Quantized Depthwise 2D Convolution with per-channel scaling.
"""
import numpy as np

def depthwise_conv2d_int8(
    input_tensor: np.ndarray,
    kernel: np.ndarray,
    scales: np.ndarray
) -> np.ndarray:
    # input_tensor: (C, H, W), kernel: (C, KH, KW)
    C, H, W = input_tensor.shape
    KC, KH, KW = kernel.shape
    pad_h, pad_w = KH // 2, KW // 2
    padded = np.pad(input_tensor, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    output = np.zeros((C, H, W), dtype=np.float32)

    for c in range(C):
        k = kernel[c]
        scale = scales[c]
        for i in range(H):
            for j in range(W):
                patch = padded[c, i:i + KH, j:j + KW]
                acc = int(np.sum(patch.astype(np.int32) * k.astype(np.int32)))
                output[c, i, j] = acc * scale
    return output
