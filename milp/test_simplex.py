import pdb
import unittest
import numpy as np
from scipy.optimize import linprog
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
        mode = "max"

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
        mode = "min"
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
        mode = "max"

        simplex = Simplex(A, b, c, const, mode)
        simplex.solve()
        exp_sol = np.array([5, 4, 0])
        self.assertEqual(simplex.obj_val(), 13)
        self.assertTrue(np.allclose(simplex.sol(), exp_sol))


    def test_add_constraint(self):
        A = np.array([
            [-1, 1],
            [1, 1]
        ])
        b = np.array([0, 2])
        c = np.array([1, 1.5])
        const = 0
        mode = "max"
        s1 = Simplex(A, b, c, const, mode)
        s1.solve()

        A2 = np.array([
                [-1, 1],
                [1, 1],
                [0, 1]
        ])
        b2 = np.array([0, 2, 0.5])
        c2 = np.array([1, 1.5])
        const2 = 0
        mode2 = "max"
        s2 = Simplex(A2, b2, c2, const2, mode2)
        s2.solve()

        s1.add_constraint(np.array([0, 1]), 0.5)
        s1.reset()
        s1.solve()

        self.assertTrue(np.allclose(s1.A, s2.A))
        self.assertTrue(np.allclose(s1.b, s2.b))
        self.assertTrue(np.allclose(s1.c, s2.c))
        self.assertTrue(np.allclose(s1.sol(), s2.sol()))
        self.assertTrue(np.allclose(s1.obj_val(), s2.obj_val()))

    def test3(self):
        '''
        minimise
            4a + 5b + 6c
        subject to
            a + b >= 11
            a - b <= 5
            c - a - b >= 0
            c - a - b <= 0
            7a >= 35 - 12b
            a >= 0 b >= 0 c >= 0
        '''
        A = np.array([
            [-1,  -1,  0],
            [ 1,  -1,  0],
            [-1,  -1,  1],
            [ 1,   1, -1],
            [-7, -12,  0],
        ])
        b = np.array([-11, 5, 0, 0, -35])
        c = np.array([4, 5, 6])
        s = Simplex(A, b, c, 0, "min")
        s.solve()
        pdb.set_trace()
        self.assertEqual(s.obj_val(), 113)
        self.assertTrue(np.allclose(s.sol(), np.array([8, 3, 11])))

if __name__ == '__main__':
    unittest.main()
