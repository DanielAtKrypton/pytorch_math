import pytorch_math as pm
import torch

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

if __name__ == "__main__":
    test_average_10()
