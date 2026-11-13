from edge_quantizer.hardware_checker import EdgeTPUValidator

def test_edge_tpu_validator():
    res_ok = EdgeTPUValidator.validate_op("Conv2D", [1, 28, 28, 3], "int8")
    assert res_ok["supported"] is True

    res_fail = EdgeTPUValidator.validate_op("CustomOp", [1, 10], "int8")
    assert res_fail["supported"] is False
