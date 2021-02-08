import unittest
import numpy as np
from gurobipy import GRB
from bc import guro_opt, bc

CUT_METHODS = ["simple", "all_gomory"]
class TestBC(unittest.TestCase):
    def test1(self):
        A = np.array([
                [-1, 2, -6],
                [1, 0, 2],
                [2, 0, 10],
                [-1, 1, 0]
        ])
        b = np.array([-10, 6, 19, -2])
        c = np.array([2, 15, 18])
        gres = guro_opt(c, A, b, vtype=GRB.INTEGER)
        for cm in CUT_METHODS:
            bsol, bobj, _ = bc(c, A, b, cm)
            self.assertTrue(np.allclose(gres.x, bsol))

    def test2(self):
        A = np.array([
            [-5, 4],
            [5, 2]
        ])
        b = np.array([0, 15])
        c = np.array([-1, -1])
        gres = guro_opt(c, A, b, vtype=GRB.INTEGER)
        for cm in CUT_METHODS:
            bsol, bobj, _ = bc(c, A, b, cm)
            self.assertTrue(np.allclose(gres.x, bsol))

    def test3(self):
        A = np.array([
            [8, 5, 3, 2],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        b = np.array([10, 1, 1, 1, 1])
        c = -np.array([15, 12, 4, 2])
        gres = guro_opt(c, A, b, vtype=GRB.INTEGER)
        for cm in CUT_METHODS:
            bsol, bobj, _ = bc(c, A, b, cm)
            self.assertTrue(np.allclose(gres.x, bsol))

    def test4(self):
        A = np.array([
            [3, 2],
            [-3, 2]
        ])
        b = np.array([6, 0])
        c = -np.array([0, 1])
        gres = guro_opt(c, A, b, vtype=GRB.INTEGER)
        for cm in CUT_METHODS:
            bsol, bobj, _ = bc(c, A, b, cm)
            self.assertTrue(np.allclose(gres.x, bsol))

if __name__ == '__main__':
    unittest.main()
