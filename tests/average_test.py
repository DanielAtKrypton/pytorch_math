"""main test
"""
import torch
from unittest import TestCase
import pytorch_math as pm
import numpy as np

class TestFailureModule(TestCase):
    """TestFailureModule
    """
    def test_average_3(self):
        """test_average_3
        """
        data = torch.arange(6).reshape((3, 2))
        weights = torch.Tensor([1./4, 3./4])
        with self.assertRaises(TypeError) as error:
            pm.average(data, weights=weights)
        self.assertEqual(
            str(error.exception),
            "Axis must be specified when shapes of a and weights differ."
        )

    def test_average_6(self):
        """test_average_6
        """
        data = torch.arange(6).reshape((3, 2))
        weights = torch.Tensor([[1, 2], [3, 4]])
        with self.assertRaises(TypeError) as error:
            pm.average(data, weights=weights, axis=1)
        self.assertEqual(
            str(error.exception),
            "1D weights expected when shapes of a and weights differ."
        )

    def test_average_8(self):
        """test_average_8
        """
        data = torch.arange(6).reshape((3, 2))
        weights = torch.Tensor([1./4, 3./4])
        with self.assertRaises(ValueError) as error:
            pm.average(data, weights=weights, axis=0)
        self.assertEqual(
            str(error.exception),
            "Length of weights not compatible with specified axis."
        )

    def test_average_9(self):
        """test_average_9
        """
        data = torch.arange(6).reshape((3, 2))
        weights = torch.Tensor([1./4, -1./4])
        with self.assertRaises(ZeroDivisionError) as error:
            pm.average(data, weights=weights, axis=1)
        self.assertEqual(
            str(error.exception),
            "Weights sum to zero, can't be normalized"
        )

def test_average_0():
    """test_average_0
    """
    X = np.random.rand(5, 5)
    avg_np, _ = np.average(X, axis=1, returned=True)

    X_th = torch.tensor(X)
    avg_th, _ = pm.average(X_th, axis=1, returned=True)

    assert (X == X_th.numpy()).all()
    assert (avg_np == avg_th.numpy()).all()

    X -= avg_np[:, None]
    X_th -= avg_th[:, None]

    assert (X == X_th.numpy()).all()

def test_average_1():
    """test_average_1
    """
    data = torch.arange(1, 5)
    assert pm.average(data) == 2.5

def test_average_2():
    """test_average_2
    """
    assert pm.average(torch.arange(1, 11), weights=torch.arange(10, 0, -1)) == 4

def test_average_4():
    """test_average_4
    """
    a = torch.ones(5, dtype=torch.int16)
    w = torch.ones(5, dtype=torch.float32)
    avg = pm.average(a, weights=w)
    assert avg.dtype == torch.float64

def test_average_5():
    """test_average_5
    """
    data = torch.arange(6).reshape((3, 2))
    result = pm.average(data, axis=1)
    expected_result = torch.Tensor([0.5, 2.5, 4.5])
    assert (result == expected_result).all()

def test_average_7():
    """test_average_7
    """
    data = torch.arange(6).reshape((3, 2))
    weights = torch.Tensor([1./4, 3./4])
    result = pm.average(data, axis=1, weights=weights)
    expected_result = torch.Tensor([0.75, 2.75, 4.75])
    assert (result == expected_result).all()

def test_average_10():
    """test_average_10
    """
    data = torch.arange(6).reshape((3, 2))
    weights = torch.Tensor([1./4, 3./4])
    result_avg, result_scl = pm.average(data, axis=1, weights=weights, returned=True)
    expected_avg = torch.Tensor([0.75, 2.75, 4.75])
    expected_scl = torch.Tensor([1., 1., 1.])
    assert (result_avg == expected_avg).all()
    assert (result_scl == expected_scl).all()

def test_average_11():
    """test_average_11
    """
    data = torch.arange(6, dtype=torch.float).reshape((3, 2))
    weights = torch.Tensor([1./4, 3./4])
    result_avg = pm.average(data, axis=1, weights=weights)
    expected_avg = torch.Tensor([0.75, 2.75, 4.75])
    assert (result_avg == expected_avg).all()

# def test_average_12():
#     """test_average_12
#     """
#     data = torch.arange(1, 5, device='cuda')
#     assert pm.average(data) == 2.5

# def test_average_13():
#     """test_average_13
#     """
#     assert pm.average(torch.arange(1, 11, device='cuda'), weights=torch.arange(10, 0, -1)) == 4

# def test_average_14():
#     """test_average_14
#     """
#     assert pm.average(torch.arange(1, 11), weights=torch.arange(10, 0, -1, device='cuda')) == 4
