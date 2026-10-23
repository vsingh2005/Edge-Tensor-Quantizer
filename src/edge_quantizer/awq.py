"""
Activation-aware Weight Quantization (AWQ) salience search.
"""
import numpy as np

class AWQSalienceSearch:
    @staticmethod
    def identify_salient_channels(activations: np.ndarray, top_k_pct: float = 0.05) -> np.ndarray:
        # activations: (Batch, Channels)
        act_magnitude = np.mean(np.abs(activations), axis=0)
        k = max(1, int(len(act_magnitude) * top_k_pct))
        top_indices = np.argsort(act_magnitude)[-k:]
        return top_indices
