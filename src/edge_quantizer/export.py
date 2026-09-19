import json
import base64
from pathlib import Path
from typing import Dict, Any
import numpy as np

class QuantizedModelExporter:
    def __init__(self, model_name: str, version: str = "1.0.0"):
        self.manifest: Dict[str, Any] = {
            "model_name": model_name,
            "version": version,
            "format": "edge_quant_v1",
            "layers": {}
        }

    def add_layer(self, layer_name: str, q_weights: np.ndarray, scale: float, zero_point: int = 0):
        self.manifest["layers"][layer_name] = {
            "shape": list(q_weights.shape),
            "dtype": str(q_weights.dtype),
            "scale": float(scale),
            "zero_point": int(zero_point),
            "weights_b64": base64.b64encode(q_weights.tobytes()).decode("ascii")
        }

    def save(self, filepath: str | Path):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    @staticmethod
    def load(filepath: str | Path) -> Dict[str, Any]:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
