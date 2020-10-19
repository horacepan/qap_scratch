import unittest
import pdb
import numpy as np
import torch

class TestGrad(unittest.TestCase):
    def test_qap(self):
        n = 10
        A = torch.rand(n, n)
        B = torch.rand(n, n)
        X = torch.rand(n, n, requires_grad=True)

        F = torch.trace(A@X@B@X.t())
        F.backward()
        expected = (A@X@B) + (A.t()@X@B.t())
        return torch.allclose(expected, X.grad)

    def test_qap_had(self):
        n = 10
        A = torch.rand(n, n)
        B = torch.rand(n, n)
        X = torch.rand(n, n, requires_grad=True)

        XX = X*X
        F = torch.trace(A@XX@B@XX.t())
        F.backward()
        expected = 2*(A@XX@B)*X + 2*(A.t()@XX@B.t())*X
        return torch.allclose(expected, X.grad)

    def test_qap_had_lagrangian(self):
        mu = np.random.random()
        n = 20
        A = torch.rand(n, n)
        B = torch.rand(n, n)
        _lambda = torch.rand((n, n))
        X = torch.rand(n, n, requires_grad=True)
        mask = (mu*X < _lambda).float()
        penalty = ((-_lambda * X) + 0.5*mu*(X*X)) * mask
        penalty = penalty + ((-_lambda * _lambda) / (2 * mu)) * (1 - mask)

        XX = X*X
        F = torch.trace(A@XX@B@XX.t()) + torch.sum(penalty)
        F.backward()
        expected = 2*(A@XX@B)*X + 2*(A.t()@XX@B.t())*X + mask *(-_lambda + (mu *X))
        return torch.allclose(expected, X.grad)

if __name__ == '__main__':
    unittest.main()
