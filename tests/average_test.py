"""main test
"""
import torch

import pytorch_math as pm

def test_average_1():
    """test_average_1
    """
    data = torch.arange(1, 5)
    assert pm.average(data) == 2.5

def test_average_2():
    """test_average_2
    """
    assert pm.average(torch.arange(1, 11), weights=torch.arange(10, 0, -1)) == 4

def test_average_3():
    """test_average_3
    """
    data = torch.arange(6).reshape((3, 2))
    weights = torch.Tensor([1./4, 3./4])
    result = pm.average(data, axis=1, weights=weights)
    expected_result = torch.Tensor([0.75, 2.75, 4.75])
    assert (result == expected_result).all()
    try:
        pm.average(data, weights=weights)
        raise Exception("Should have raised an exception!")
    # pylint: disable=broad-except
    except Exception as exception:
        assert type(exception).__name__ == 'TypeError'
        assert str(exception) == "Axis must be specified when shapes of a and weights differ."

def test_average_4():
    """test_average_4
    """
    a = torch.ones(5, dtype=torch.int16)
    w = torch.ones(5, dtype=torch.float32)
    avg = pm.average(a, weights=w)
    assert avg.dtype == torch.float64
