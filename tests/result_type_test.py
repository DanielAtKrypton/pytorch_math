"""result_type_test.py
"""
import torch
from pytorch_math import result_type

def test_result_type_1():
    """test_result_type_1
    """
    test_array = [
        torch.Tensor([0]).type(torch.int),
        torch.Tensor([1, 2, 4]).type(torch.int8),
        torch.Tensor([0]).type(torch.float64)
    ]
    assert result_type(test_array) == torch.float64
