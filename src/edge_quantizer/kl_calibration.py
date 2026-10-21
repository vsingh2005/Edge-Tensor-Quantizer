"""
TensorRT-style KL-divergence histogram calibration.
"""
import numpy as np
from scipy.stats import entropy

class KLHistogramCalibrator:
    @staticmethod
    def calibrate_threshold(data: np.ndarray, num_bins: int = 2048, target_bins: int = 128) -> float:
        flat = np.abs(data.flatten())
        max_val = float(np.max(flat))
        if max_val == 0:
            return 1.0
            
        hist, bin_edges = np.histogram(flat, bins=num_bins, range=(0, max_val))
        best_kl = float("inf")
        best_threshold = max_val

        # Sample thresholds between target_bins and num_bins
        for t_idx in range(target_bins, min(target_bins + 200, num_bins), 10):
            p = hist[:t_idx].astype(np.float64)
            p[-1] += np.sum(hist[t_idx:])
            if np.sum(p) == 0:
                continue
            p /= np.sum(p)

            # Quantize down to target_bins
            chunk_size = t_idx / target_bins
            q = np.array([np.sum(p[int(i * chunk_size):int((i + 1) * chunk_size)]) for i in range(target_bins)])
            # Expand back
            expanded_q = np.repeat(q / chunk_size, int(np.ceil(chunk_size)))[:t_idx]
            if np.sum(expanded_q) == 0:
                continue
            expanded_q /= np.sum(expanded_q)

            kl = entropy(p + 1e-10, expanded_q + 1e-10)
            if kl < best_kl:
                best_kl = kl
                best_threshold = float(bin_edges[t_idx])

        return best_threshold
