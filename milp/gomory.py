import pdb
import numpy as np
from scipy.optimize import linprog

from simplex import Simplex

def gomory_cuts(tableau, A_orig, b_orig):
    '''
    tableau: A,b stack c, const
             A = m x n
             b = m
             c = n
             e = n (num vars)
             r = m (number of slacks)
             ncols = num vars
    '''
    ncols = A_orig.shape[1]
    A_t = tableau[:-1, :-1]
    b_t = tableau[:-1, -1]

    idxs = ~np.equal(np.mod(b_t, 1), 0)
    At =     -A_t[idxs] + np.floor(A_t[idxs])
    d = bt = -b_t[idxs] + np.floor(b_t[idxs])

    e = At[:, :ncols] # k x n
    r = At[:, ncols:] # k x n
    try:
        lhs = e - r@A_orig  # (k x n) - (k x m) x (m x n) = k x n
        rhs = d - r@b_orig # k - (k x m) x m x 1
    except:
        pdb.set_trace()
    return lhs, rhs

def pick_gomory_cut(At, bx):
    idx = np.random.randint(0, At.shape[0])
    print("new cut", np.round(At[idx], 10), np.round(bx[idx], 10))
    pdb.set_trace()
    return At[idx], bx[idx]

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
        self.A_orig = A.copy()
        self.b_orig = b.copy()
        self.ncols = A.shape[1]

    def solve(self):
        niters = 0
        tableau = []
        sol = None
        obj = 0

        def log_tableau(x, **kwargs):
            tableau.append(kwargs["tableau"])

        while sol is None or  ~np.equal(np.mod(sol, 1), 0):
            res = linprog(self.c, self.A, self.b, callback=log_tableau)
            niters += 1

            if np.all(np.equal(np.mod(res.x, 1), 0)):
                sol = res.x
                obj = res.fun
                break
            else:
                tab = tableau[-1]
                A_cuts, b_cuts = gomory_cuts(tab, self.A, self.b)
                Ai, bi = pick_gomory_cut(A_cuts, b_cuts)
                Ai = np.round(Ai, 10)
                bi = np.round(bi, 10)
                self.A = np.vstack([self.A, Ai])
                self.b = np.append(self.b, bi)
        pdb.set_trace()
        return sol, obj

if __name__ == '__main__':
    A = np.array([
        [-5, 4],
        [5, 2],
    ])
    b = np.array([
        0, 15#, 0, 0, 2, 2
    ])
    c = -np.array([1, 1])
    res = linprog(c, A, b)
    print('LP relaxation solution:', res.x)

    g = Gomory(A, b, c)
    sol, obj = g.solve()
    print(sol)
    print(obj)
    print(A @ np.array([2, 2]) <= b)
    print(c @ np.array([2, 2]))
