import pdb
import time
from cvxopt import solvers
import numpy as np
from collections import deque, namedtuple
from scipy.optimize import linprog
from gurobipy import Model, GRB, quicksum

Result = namedtuple("Result", ("x", "fun", "success"))

def guro_opt(c, A, b, minimize=True):
    m = Model()
    m.setParam('OutputFlag', 0)
    mode = GRB.MINIMIZE if minimize else GRB.MAXIMIZE
    _vars = [m.addVar(lb=0, vtype="C") for i in range(A.shape[1])]
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
        cuts.append((-ai, bi_ceil, xids))

    return cuts

def gen_gomory_cuts(tableau, A_orig, b_orig):
    ncols = A.shape[1]
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
    return [(lhs[i], rhs[i], None) for i in range(lhs.shape[0])]

def add_cut(A, b, ai, bi):
    ai = np.round(ai, 8)
    bi = np.round(bi, 8)
    A_new = np.vstack([A, ai])
    b_new = np.append(b, bi)
    return A_new, b_new

def get_next_node(q):
    return q.popleft()

def is_int(x):
    return np.all(np.equal(np.mod(x, 1), 0))

def bc(A, b, c, cut_method="simple"):
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
        #lp_sol = guro_opt(c, curr_A, curr_b)
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
            for a_cut, b_cut, xids in cuts:
                A_new, b_new = add_cut(curr_A, curr_b, a_cut, b_cut)
                q.append((A_new, b_new, xids))

    print("Nodes explored: {}".format(nnodes))
    return opt_sol, opt_obj

if __name__ == '__main__':
    print('Gurobi check')
    m = Model()
    m.setParam('OutputFlag', 0)
    x = m.addVar(lb=0, vtype='I')
    y = m.addVar(lb=0, vtype='I')
    m.addConstr(-5*x + 4*y <= 0)
    m.addConstr( 5*x + 2*y <= 15)
    m.setObjective(-x - y, GRB.MINIMIZE)
    m.optimize()
    print("sol: [{}, {}]".format(x.X, y.X))
    print("obj:", m.objVal)
    print('====================')

    '''
    15x1+ 12x2+ 4x3+ 2x4
    s.t.8x1+ 5x2+ 3x3+ 2x4≤10
    xi∈{0,1}
    '''
    st = time.time()
    A = np.array([
        [8, 5, 3, 2],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    b = np.array([10, 1, 1, 1, 1])
    c = -np.array([15, 12, 4, 2])

    print("bc:")
    #A = np.array([
    #    [-5, 4],
    #    [5, 2]
    #])
    #b = np.array([0, 15])
    #c = np.array([-1, -1])

    sol, obj = bc(A, b, c, "simple")
    print('sol:', sol)
    print('obj:', obj)
    print('====================')
    print("Elapsed: {:.2f}s".format(time.time() -st))
    print('===============')

    st = time.time()
    m = Model()
    m.setParam("OutputFlag", 0)
    _vars = []
    for _ in range(A.shape[1]):
        _vars.append(m.addVar(lb=0, ub=1, vtype='I'))
    for idx in range(A.shape[1]):
        row = A[idx]
        m.addConstr(quicksum(row[i] * _vars[i] for i in range(A.shape[1])) <= b[idx])
    m.setObjective(quicksum(_vars[i] * c[i] for i in range(A.shape[1])), GRB.MINIMIZE)
    print("setup  elapsed:", time.time() - st)
    m.optimize()
    sol = [v.X for v in _vars]
    print(m.objVal)
    print("gurobi sol:", sol)
    print("gurobi elapsed:", time.time() - st)
