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
        self.assertTrue(torch.allclose(expected, X.grad))

    def test_qap_had(self):
        n = 10
        A = torch.rand(n, n)
        B = torch.rand(n, n)
        X = torch.rand(n, n, requires_grad=True)

        XX = X*X
        F = torch.trace(A@XX@B@XX.t())
        F.backward()
        expected = 2*(A@XX@B)*X + 2*(A.t()@XX@B.t())*X
        self.assertTrue(torch.allclose(expected, X.grad))

    def test_qap_had_lagrangian(self):
        mu = np.random.random()
        n = 10
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
        self.assertTrue(torch.allclose(expected, X.grad))


    def test_xaxt(self):
        n = 10
        A = torch.rand(n, n)
        B = torch.rand(n, n)
        X = torch.rand(n, n, requires_grad=True)
        grad = A@X@B + A.t()@X@B.t()

        F = torch.trace(A@X@B@X.t())
        F.backward()
        self.assertTrue(torch.allclose(grad, X.grad))

    def test_xtax(self):
        n = 10
        A = torch.rand(n, n)
        B = torch.rand(n, n)
        X = torch.rand(n, n, requires_grad=True)
        grad = B@X@A + B.t()@X@A.t()

        F = torch.trace(A@X.t()@B@X)
        F.backward()
        self.assertTrue(torch.allclose(grad, X.grad))

    def test_frob_orth(self):
        n = 10
        A = torch.rand(n, n)
        B = torch.rand(n, n)
        X = torch.rand(n, n)
        X, _ = torch.qr(X) # make an orthogonal x
        X.requires_grad = True

        #ftrace= torch.trace((A - X@B@X.t()).t() @ (A - X@B@X.t()))
        #fnorm= torch.norm(A - X@B@X.t())**2
        f_part =  -torch.trace(A.t() @ X@B@X.t()) - torch.trace(X@B.t()@X.t() @ A) + torch.trace(X@B.t()@B@X.t())
        f_part.backward()

        grad = -2 * (A.t() @ X @ B  + A @ X @ B.t()) + 2*(X @ B.t() @ B)
        self.assertTrue(torch.allclose(X.grad, grad, atol=1e-6))

if __name__ == '__main__':
    unittest.main()
