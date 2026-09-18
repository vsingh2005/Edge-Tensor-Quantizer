import numpy as np

class CalibrationEngine:
    @staticmethod
    def find_optimal_threshold_mse(tensor: np.ndarray, num_bins: int = 128) -> float:
        flat = np.abs(tensor.flatten())
        max_val = float(np.max(flat))
        if max_val == 0.0:
            return 1.0
        best_threshold = max_val
        min_mse = float("inf")
        for candidate in np.linspace(0.70 * max_val, max_val, num_bins):
            scale = candidate / 127.0
            clipped = np.clip(tensor, -candidate, candidate)
            q = np.clip(np.round(clipped / scale), -128, 127)
            dequant = q * scale
            mse = float(np.mean((tensor - dequant) ** 2))
            if mse < min_mse:
                min_mse = mse
                best_threshold = float(candidate)
        return best_threshold

    @staticmethod
    def calibrate_weights_per_channel(weights: np.ndarray) -> np.ndarray:
        out_channels = weights.shape[0]
        thresholds = np.zeros(out_channels, dtype=np.float32)
        for i in range(out_channels):
            thresholds[i] = CalibrationEngine.find_optimal_threshold_mse(weights[i])
        return thresholds
