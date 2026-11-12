"""
GGUF binary format metadata header serializer.
"""
import struct
from typing import List, Dict, Any

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian

class GGUFSerializer:
    @staticmethod
    def serialize_header(tensors: List[Dict[str, Any]], version: int = 3) -> bytes:
        tensor_count = len(tensors)
        metadata_kv_count = 0
        header = struct.pack("<IIQQ", GGUF_MAGIC, version, tensor_count, metadata_kv_count)
        return header
