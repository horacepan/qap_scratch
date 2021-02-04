import pdb
import numpy as np
from collections import namedtuple
from scipy.optimize import linprog
from gomory import gomory_cuts
from bc import gen_simple_cuts, guro_opt

def generate_max_cut(n, e):
    c = np.zeros(n + e)
    c[n:] = -1

    all_edges = [(i, j) for i in range(n) for j in range(i)]
    picked_idx = np.random.choice(len(all_edges), size=e, replace=False)
    edges = [all_edges[i] for i in picked_idx]
    A = np.zeros((3*e, n + e))
    b = np.zeros(3*e)
    row = 0

    for idx, (u, v) in enumerate(edges):
        e_idx = n + idx
        b[row] = 0
        A[row, e_idx] = 1
        A[row, u] = -1
        A[row, v] = -1

        b[row + 1] = 2
        A[row + 1, u] = 1
        A[row + 1, v] = 1
        A[row + 1, e_idx] = 1

        A[row + 2, e_idx] = 1
        b[row + 2] = 1
        row += 3

    return A, b, c

class EnvMILP:
    def __init__(self, c, A, b):
        self.A = A
        self.b = b
        self.c = c

    def state(self):
        A, b = self.simplex.constraints()
        c = self.simplex.objective()
        D = self.gomory_cuts(A, b)
        return A, b, c, D

    def actions(self):
        return self.MILP.gomory_cuts()

    def step(self, action):
        cut = self._gomory_cuts[idx]
        self.add_cut(cut)

    def reset(self):
        pass

MaxCutState = namedtuple("MaxCutState", ("A", "b", "c", "curr_sol", "cuts"))

class MaxCutMILP:
    def __init__(self, num_verts, num_edges, cut_type='gomory'):
        self.num_verts = num_verts
        self.num_edges = num_edges
        self.cut_type = cut_type

        self._tableau = [None]
        self.curr_sol = None
        self.curr_obj = None
        self.prev_obj = None

        self.A = None
        self.b = None
        self.c = None

    def log_tableau(self, x, **kwargs):
        self._tableau[0] = kwargs["tableau"]

    def reset(self):
        A, b, c = generate_max_cut(self.num_verts, self.num_edges)
        self._tableau = [None]
        res = linprog(c, A, b, callback=self.log_tableau)
        self.curr_obj = res.fun
        self.curr_sol = res.x
        self.A = A
        self.b = b
        self.c = c
        self.curr_cuts = gomory_cuts(self._tableau[0], A, b)

        return MaxCutState(self.A, self.b, self.c, self.curr_sol, self.curr_cuts)

    def actions(self):
        '''
        return ai, bi pair?
        '''
        if self.cut_type == 'gomory':
            ncols = self.A.shape[1]
            A_t = self._tableau[0][:-1, :-1]
            b_t = self._tableau[0][:-1, -1]

            idxs = ~np.equal(np.mod(b_t, 1), 0)
            At =     -A_t[idxs] + np.floor(A_t[idxs])
            d = bt = -b_t[idxs] + np.floor(b_t[idxs])

            e = At[:, :ncols] # k x n
            r = At[:, ncols:] # k x n
            lhs = e - r@self.A  # (k x n) - (k x m) x (m x n) = k x n
            rhs = d - r@self.b # k - (k x m) x m x 1
            return lhs, rhs
        else:
            raise Exception("Not implemented")

    # take this as an int or a vector?
    def step(self, action):
        # add the stuff to A, B, c?
        Ai, bi = action
        A_new = np.vstack([self.A, Ai])
        b_new = np.append(self.b, bi)
        self.A = A_new
        self.b = b_new

        curr_lp_sol = linprog(c, self.A, self.b, callback=self.log_tableau)
        self.prev_obj = self.curr_obj
        self.curr_sol = curr_lp_sol.x
        self.curr_obj = curr_lp_sol.fun
        self.curr_cuts = self.actions()

        done = np.all(np.equal(np.mod(curr_lp_sol.x, 1), 0))
        reward = self.prev_obj - self.curr_obj # for minimization task, curr < prev
        return MaxCutState(self.A, self.b, self.c, self.curr_sol, self.curr_cuts), reward, done, None

if __name__ == '__main__':
    A, b, c = generate_max_cut(5, 10)
    res = guro_opt(c, A, b, minimize=False, vtype='I')
    print(A.shape, b.shape, c.shape)
    print(res)
    print('===========')

    num_verts = 7
    num_edges = 20
    env = MaxCutMILP(num_verts, num_edges)
    A, b, c, sol, cuts = env.reset()
    print(env._tableau)
    A_cut, b_cut = env.actions()
    print(A_cut)
    print(A_cut.shape, b_cut.shape)
    ix = 0
    env.step((A_cut[ix], b_cut[ix]))
