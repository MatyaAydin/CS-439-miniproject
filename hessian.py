import torch
import torch.nn as nn
import unittest
from torch.func import functional_call
from torch.autograd.functional import hessian

# The compute_hessian function
def compute_hessian(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: callable) -> torch.Tensor:
    params = dict(model.named_parameters())
    param_shapes = [(name, p.shape) for name, p in params.items()]
    flat_params = torch.cat([p.detach().reshape(-1) for p in params.values()]).requires_grad_(True)

    def unflatten_params(flat_params):
        param_dict = {}
        idx = 0
        for name, shape in param_shapes:
            n = torch.tensor(shape).prod().item()
            param_dict[name] = flat_params[idx:idx+n].view(shape)
            idx += n
        return param_dict

    def wrapped_loss(flat_params):
        new_params = unflatten_params(flat_params)
        y_pred = functional_call(model, new_params, (x,))
        return loss_fn(y_pred, y)

    return hessian(wrapped_loss, flat_params)

# Linear model: f(x) = wx + b
class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=True)

    def forward(self, x):
        return self.linear(x)

# Quadratic model: f(x) = ax^2 + b
class QuadraticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([0.0]))
        self.b = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        return self.a * x**2 + self.b

# Unit test class
class TestHessianComputation(unittest.TestCase):
    def test_linear_model(self):
        x_val = 3.0
        x = torch.tensor([[x_val]])
        y = torch.tensor([[1.0]])

        model = SimpleLinear()
        with torch.no_grad():
            model.linear.weight.fill_(0.0)
            model.linear.bias.fill_(0.0)

        def loss_fn(y_pred, y_true):
            return 0.5 * ((y_pred - y_true) ** 2).mean()

        H = compute_hessian(model, x, y, loss_fn)
        expected = torch.tensor([
            [x_val**2, x_val],
            [x_val, 1.0]
        ])
        H2x2 = H[:2, :2]
        self.assertTrue(torch.allclose(H2x2, expected, atol=1e-6),
                        f"Linear model Hessian mismatch:\nExpected:\n{expected}\nGot:\n{H2x2}")

    def test_quadratic_model(self):
        x_val = 2.0
        x = torch.tensor([[x_val]])
        y = torch.tensor([[1.0]])

        model = QuadraticModel()
        with torch.no_grad():
            model.a.fill_(0.0)
            model.b.fill_(0.0)

        def loss_fn(y_pred, y_true):
            return 0.5 * ((y_pred - y_true) ** 2).mean()

        H = compute_hessian(model, x, y, loss_fn)
        x2 = x_val ** 2
        x4 = x_val ** 4
        expected = torch.tensor([
            [x4, x2],
            [x2, 1.0]
        ])
        H2x2 = H[:2, :2]
        self.assertTrue(torch.allclose(H2x2, expected, atol=1e-6),
                        f"Quadratic model Hessian mismatch:\nExpected:\n{expected}\nGot:\n{H2x2}")

if __name__ == "__main__":
    unittest.main()
