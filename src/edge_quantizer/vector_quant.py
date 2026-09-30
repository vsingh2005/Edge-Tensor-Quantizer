"""
Vector quantization with k-means codebook generation.
"""
import numpy as np
from typing import Tuple

class VectorQuantizer:
    def __init__(self, num_centroids: int = 16, max_iter: int = 20):
        self.k = num_centroids
        self.max_iter = max_iter
        self.codebook = None

    def fit(self, data: np.ndarray, seed: int = 42):
        rng = np.random.RandomState(seed)
        flat = data.flatten()
        centroids = rng.choice(flat, size=self.k, replace=False)
        for _ in range(self.max_iter):
            dists = np.abs(flat[:, None] - centroids[None, :])
            labels = np.argmin(dists, axis=1)
            new_centroids = np.array([flat[labels == i].mean() if np.any(labels == i) else centroids[i] for i in range(self.k)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        self.codebook = np.sort(centroids)

    def encode(self, data: np.ndarray) -> np.ndarray:
        flat = data.flatten()
        dists = np.abs(flat[:, None] - self.codebook[None, :])
        return np.argmin(dists, axis=1).reshape(data.shape).astype(np.uint8)

    def decode(self, indices: np.ndarray) -> np.ndarray:
        return self.codebook[indices]
