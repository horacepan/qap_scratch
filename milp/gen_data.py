import numpy as np

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
