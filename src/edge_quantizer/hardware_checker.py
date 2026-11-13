"""
Edge TPU op-compatibility and tensor shape constraints validator.
"""
from typing import Dict, Any

SUPPORTED_OPS = {"Conv2D", "DepthwiseConv2D", "FullyConnected", "MaxPool2D", "Add", "Softmax"}

class EdgeTPUValidator:
    @staticmethod
    def validate_op(op_type: str, input_shape: list, dtype: str) -> Dict[str, Any]:
        if op_type not in SUPPORTED_OPS:
            return {"supported": False, "reason": f"Op {op_type} not supported by Edge TPU compiler"}
        if dtype not in ["int8", "uint8"]:
            return {"supported": False, "reason": f"Dtype {dtype} unsupported; must be INT8/UINT8"}
        # Outer dimension check
        if len(input_shape) > 4:
            return {"supported": False, "reason": "Tensors rank > 4 unsupported"}
        return {"supported": True, "reason": "Fully compatible"}
