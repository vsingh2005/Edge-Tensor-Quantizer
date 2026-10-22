import numpy as np
from edge_quantizer.gptq import GPTQCompensationSolver

def test_gptq_solver():
    w = np.array([1.2, -3.4, 2.1, 0.5], dtype=np.float32)
    H_inv = np.eye(4, dtype=np.float32)
    q_w = GPTQCompensationSolver.quantize_row(w, H_inv, scale=0.5)
    assert len(q_w) == 4
