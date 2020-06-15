import torch
import pytorch_math as pm

def test_average_6():
    """test_average_6
    """
    data = torch.arange(6).reshape((3, 2))
    weights = torch.Tensor([[1, 2], [3, 4]])
    try:
        pm.average(data, weights=weights, axis=1)
        raise Exception("Should have raised an exception!")
    # pylint: disable=broad-except
    except Exception as exception:
        assert type(exception).__name__ == 'TypeError'
        assert str(exception) == "1D weights expected when shapes of a and weights differ."

if __name__ == "__main__":
    test_average_6()
    pass
