import numpy as np
from edge_quantizer.evaluator import QuantizationEvaluator

def test_sqnr_evaluation():
    orig = np.array([1.0, 2.0, 3.0, 4.0])
    noisy = orig + 0.01
    sqnr = QuantizationEvaluator.compute_sqnr(orig, noisy)
    assert sqnr > 30.0
    sim = QuantizationEvaluator.compute_cosine_similarity(orig, noisy)
    assert sim > 0.99
