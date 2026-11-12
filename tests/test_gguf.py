from edge_quantizer.gguf import GGUFSerializer

def test_gguf_serializer():
    header = GGUFSerializer.serialize_header([{"name": "tensor1"}])
    assert len(header) == 24
    assert header.startswith(b"GGUF")
