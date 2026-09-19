import tempfile
from pathlib import Path
import numpy as np
from edge_quantizer.export import QuantizedModelExporter

def test_quant_model_export_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / "model_quant.json"
        exporter = QuantizedModelExporter(model_name="test_conv", version="0.1")
        w = np.array([[1, 2], [3, 4]], dtype=np.int8)
        exporter.add_layer("conv1.weight", w, scale=0.05)
        exporter.save(out_file)
        loaded = QuantizedModelExporter.load(out_file)
        assert loaded["model_name"] == "test_conv"
        assert "conv1.weight" in loaded["layers"]
        assert loaded["layers"]["conv1.weight"]["scale"] == 0.05
