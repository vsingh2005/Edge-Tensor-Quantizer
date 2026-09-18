import numpy as np

class FusedDenseKernel:
    @staticmethod
    def forward(
        x_int8: np.ndarray,
        w_int8: np.ndarray,
        bias: np.ndarray,
        x_scale: float,
        w_scale: float,
        out_scale: float,
        apply_relu: bool = True
    ) -> np.ndarray:
        accum_int32 = np.dot(x_int8.astype(np.int32), w_int8.astype(np.int32))
        combined_scale = (x_scale * w_scale) / out_scale
        scaled_accum = accum_int32.astype(np.float32) * combined_scale + (bias / out_scale)
        if apply_relu:
            scaled_accum = np.maximum(0.0, scaled_accum)
        out_int8 = np.clip(np.round(scaled_accum), -128, 127).astype(np.int8)
        return out_int8
