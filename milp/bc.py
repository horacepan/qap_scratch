import numpy as np
from collections import deque
from scipy.optimize import linprog

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

def add_cut(A, b, ai, bi):
    A_new = np.vstack([A, ai])
    b_new = np.append(b, bi)
    return A_new, b_new

def get_next_node(q):
    return q.pop()

def is_int(x):
    return np.all(np.equal(np.mod(x, 1), 0))

def bc(A, b, c):
    node = (A, b, set())
    q = deque()
    q.append(node)
    opt_lb = float('inf')
    opt_obj = float('inf')
    opt_sol = None
    nnodes = 0

    while len(q) > 0:
        curr_A, curr_b, xids = get_next_node(q)
        lp_sol = linprog(c, curr_A, curr_b)
        nnodes += 1

        if not lp_sol.success:
            continue
        if is_int(lp_sol.x) and lp_sol.fun <= opt_obj:
            opt_obj = lp_sol.fun
            opt_sol = lp_sol.x
            continue
        else:
            cuts = gen_simple_cuts(lp_sol.x, xids)
            for a_cut, b_cut, xids in cuts:
                A_new, b_new = add_cut(curr_A, curr_b, a_cut, b_cut)
                q.append((A_new, b_new, xids))

    return opt_sol, opt_obj

if __name__ == '__main__':
    A = np.array([
        [-5, 4],
        [5, 2]
    ])
    b = np.array([
        0, 15
    ])
    c = np.array([
        -1, -1
    ])
    sol, obj = bc(A, b, c)
    print('sol:', sol)
    print('obj:', obj)
