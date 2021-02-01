import pdb
import numpy as np
from scipy.optimize import linprog

from simplex import Simplex

def gomory_cuts(tableau, A_orig, b_orig, ncols):
    '''
    tableau: A,b stack c, const
             A = m x n
             b = m
             c = n
             e = n (num vars)
             r = m (number of slacks)
             ncols = num vars
    '''
    A_t = tableau[:-1, :-1]
    b_t = tableau[:-1, -1]

    idxs = np.equal(np.mod(b_t, 1), 0)
    At = -A_t[idxs] + np.floor(A_t[idxs])
    d = bt = -b_t[idxs] + np.floor(b_t[idxs])

    e = At[:, :ncols] # k x n
    r = At[:, ncols:] # k x n
    lhs = e - r@A_orig  # (k x n) - (k x m) x (m x n) = k x n
    rhs = d - r@b_orig # k - (k x m) x m x 1
    return lhs, rhs

class Gomory:
    def __init__(self, A, b, c, const=0, mode="min"):
        '''
        Solve:
        min c^\top x
        subject to:
            Ax \leq b
        '''

        self.A = A
        self.b = b
        self.c = c

    def solve(self):
        niters = 0
        tableau = [None]
        def log_tableau(x, **kwargs):
            tableau[0] = kwargs["tableau"]

        while niters < 100:
            res = linprog(c, A, b, callback=log_tableau)
            niters += 1

            if np.equal(np.mod(res.x, 1), 0):
                break
            else:
                tab = tableau[0]
                A_t = tab[:-1, :-1]
                b_t = tab[:-1, :-1]
                c_t = tab[-1, :-1]
                const = tab[-1, -1]

                A_cuts, b_cuts = gomory_cuts(A_t, b_t)

        obj = self.simplex.obj_val()
        sol = self.simplex.sol()
        return obj_val, sol

if __name__ == '__main__':
    m = 4
    n = 5
    A = np.random.random((m, n))
    b = np.random.random(m)
    b[0] = 1
    b[1] = 2
    b[2] = 5
    c = np.random.random(n+m)
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :n] = A
    tableau[:m, n:-1] = np.eye(m)
    tableau[:-1, -1] = b
    tableau[-1, :-1] = c
    print(tableau)
    At, bt = gomory_cuts(tableau, A, b, n)
    print(At.shape, bt.shape)
