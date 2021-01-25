import unittest
import numpy as np
from simplex import Simplex

class TestSimplex(unittest.TestCase):
    def test_max(self):
        A = np.array([
            [1, 0, 0],
            [2, 1, 1],
            [2, 2, 1],
        ])
        b = np.array([4, 10, 16])
        c = np.array([20, 16, 12])
        const = 10
        mode = "maximize"

        simplex = Simplex(A, b, c, const, mode)
        simplex.solve()
        exp_sol = np.array([0, 6, 4])
        self.assertEqual(simplex.obj_val(), 154)
        self.assertTrue(np.allclose(simplex.sol(), exp_sol))

    def test_min(self):
        A = np.array([
            [ 1,  0, 0],
            [ 0,  1, 0],
            [ 1,  1, 0],
            [-1,  0, 2],
        ])
        b = np.array([4, 4, 6, 4])
        c = np.array([-1, 2, -1])
        const = 0
        mode = "minimize"
        simplex = Simplex(A, b, c, const, mode)
        simplex.solve()
        exp_sol = np.array([4, 0, 4])
        self.assertEqual(simplex.obj_val(), -8)
        self.assertTrue(np.allclose(simplex.sol(), exp_sol))

    def test_max2(self):
        A = np.array([
            [2, 1, 1],
            [4, 2, 3],
            [2, 5, 5],
        ])
        b = np.array([14, 28, 30])
        c = np.array([1, 2, -1])
        const = 0
        mode = "maximize"

        simplex = Simplex(A, b, c, const, mode)
        simplex.solve()
        exp_sol = np.array([5, 4, 0])
        self.assertEqual(simplex.obj_val(), 13)
        self.assertTrue(np.allclose(simplex.sol(), exp_sol))

if __name__ == '__main__':
    unittest.main()
