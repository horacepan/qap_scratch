import pdb
import time
from cvxopt import solvers
import numpy as np
from collections import deque, namedtuple
from scipy.optimize import linprog
from gurobipy import Model, GRB, quicksum

Result = namedtuple("Result", ("x", "fun", "success"))

def guro_opt(c, A, b, minimize=True, vtype=GRB.CONTINUOUS):
    m = Model()
    m.setParam('OutputFlag', 0)
    mode = GRB.MINIMIZE if minimize else GRB.MAXIMIZE
    _vars = [m.addVar(lb=0, vtype=vtype) for i in range(A.shape[1])]
    m.setObjective(quicksum(_vars[i]*c[i] for i in range(len(c))), mode)

    for idx in range(A.shape[0]):
        row = A[idx]
        m.addConstr(quicksum(row[i]*_vars[i] for i in range(A.shape[1])) <= b[idx])

    m.optimize()
    results = {
        "status": m.status,
        "obj": m.objVal,
        "sol": np.array([v.X for v in _vars])
    }
    res_tup = Result(results['sol'], m.objVal, m.status == GRB.OPTIMAL)
    return res_tup

def gen_simple_cuts(x, excluded_indices=None):
    if excluded_indices is None:
        excluded_indices = set()

    cuts = []
    for i in range(len(x)):
        if i in excluded_indices:
            continue

        ai = np.zeros(x.shape)
        ai[i] = 1
        bi_ceil = np.ceil(x[i])
        bi_floor = np.floor(x[i])

        xids = excluded_indices.union([i])
        cuts.append((ai, bi_floor, xids))
        cuts.append((-ai, -bi_ceil, xids))

    return cuts

def gen_gomory_cuts(tableau, A_orig, b_orig):
    ncols = A_orig.shape[1]
    A_t = tableau[:-1, :-1]
    b_t = tableau[:-1, -1]

    idxs = ~np.equal(np.mod(b_t, 1), 0)
    At =     -A_t[idxs] + np.floor(A_t[idxs])
    d = bt = -b_t[idxs] + np.floor(b_t[idxs])

    e = At[:, :ncols] # k x n
    r = At[:, ncols:] # k x n
    lhs = e - r@A_orig  # (k x n) - (k x m) x (m x n) = k x n
    rhs = d - r@b_orig # k - (k x m) x m x 1
    return [(lhs[i], rhs[i], None) for i in range(lhs.shape[0])]

def gen_all_gomory_cuts(tableau, A_orig, b_orig):
    ncols = A_orig.shape[1]
    A_t = tableau[:-1, :-1]
    b_t = tableau[:-1, -1]

    idxs = ~np.equal(np.mod(b_t, 1), 0)
    At =     -A_t[idxs] + np.floor(A_t[idxs])
    d = bt = -b_t[idxs] + np.floor(b_t[idxs])

    e = At[:, :ncols] # k x n
    r = At[:, ncols:] # k x n
    lhs = e - r@A_orig  # (k x n) - (k x m) x (m x n) = k x n
    rhs = d - r@b_orig # k - (k x m) x m x 1

    unique_lhs, unique_idx = np.unique(lhs, return_index=True, axis=0)
    unique_rhs = rhs[unique_idx]
    return unique_lhs, unique_rhs

def add_cut(A, b, ai, bi):
    ai = np.round(ai, 8)
    bi = np.round(bi, 8)
    A_new = np.vstack([A, ai])
    b_new = np.append(b, bi)
    return A_new, b_new

def get_next_node(q):
    return q.popleft()

def is_int(x):
    rounded_x = np.round(x, 8)
    return np.all(np.equal(np.mod(rounded_x, 1), 0))

def bc(c, A, b, cut_method="simple", debug_true_sol=None):
    node = (A, b, set())
    q = deque()
    q.append(node)
    opt_lb = float('inf')
    opt_obj = float('inf')
    opt_sol = None
    nnodes = 0
    tableau = [None]

    def log_tableau(x, **kwargs):
        tableau[0] = kwargs["tableau"]

    while len(q) > 0:
        curr_A, curr_b, xids = get_next_node(q)
        lp_sol = linprog(c, curr_A, curr_b, callback=log_tableau)
        nnodes += 1

        if not lp_sol.success:
            continue
        if is_int(lp_sol.x) and lp_sol.fun <= opt_obj:
            opt_obj = lp_sol.fun
            opt_sol = lp_sol.x
            continue
        else:
            if cut_method == "simple":
                cuts = gen_simple_cuts(lp_sol.x, xids)
            elif cut_method == "gomory":
                cuts = gen_gomory_cuts(tableau[0], curr_A, curr_b)
            elif cut_method == "all_gomory":
                A_cut, b_cut = gen_all_gomory_cuts(tableau[0], curr_A, curr_b)
                A_new, b_new = add_cut(curr_A, curr_b, A_cut, b_cut)
                q.append((A_new, b_new, xids))
                #if debug_true_sol is not None:
                #    valid = A_new @ debug_true_sol <= b_new
                #    if not np.all(valid):
                #        raise Exception("Gomory cuts invalid | Numerical error?")
                continue

            for a_cut, b_cut, xids in cuts:
                A_new, b_new = add_cut(curr_A, curr_b, a_cut, b_cut)
                q.append((A_new, b_new, xids))

    return opt_sol, opt_obj, nnodes

if __name__ == '__main__':
    np.random.seed(0)
    cut_method = "all_gomory"
    A = np.array([
            [-1, 2, -6],
            [1, 0, 2],
            [2, 0, 10],
            [-1, 1, 0]
    ])
    b = np.array([-10, 6, 19, -2])
    c = np.array([2, 15, 18])
    print('gurobi', guro_opt(c, A, b, vtype=GRB.INTEGER))
    print('bc:', bc(c, A, b, cut_method))
    print('=====')

    A = np.array([
        [-5, 4],
        [5, 2]
    ])
    b = np.array([0, 15])
    c = np.array([-1, -1])
    print("gurobi", guro_opt(c, A, b, vtype=GRB.INTEGER))
    print("bc", bc(c, A, b, cut_method))
    print('=====')

    A = np.array([
        [8, 5, 3, 2],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    b = np.array([10, 1, 1, 1, 1])
    c = -np.array([15, 12, 4, 2])
    print("gurobi", guro_opt(c, A, b, vtype=GRB.INTEGER))
    print("bc", bc(c, A, b, cut_method))
    print('=====')

    A = np.array([
        [3, 2],
        [-3, 2]
    ])
    b = np.array([6, 0])
    c = -np.array([0, 1])
    print("gurobi", guro_opt(c, A, b, vtype=GRB.INTEGER))
    print("bc", bc(c, A, b, cut_method))
