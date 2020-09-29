import pdb
import argparse
import numpy as np
from admm import admm_qap
from mat_utils import brute_solve

def sdiff(s1, s2):
    return [i for i in s1 if i not in s2]

def bb_admm(A, B, C, ub, X, i, j, args):
    '''
    A: flow matrix
    B: flow matrix
    C: linear cost matrix
    ub: current upper bound
    X: existing assignment
    i: facility being assigned
    j: location being assigned to
    args: namespace of args
    '''
    n = A.shape[0]
    els = [i for i in range(n)]
    rows, cols = np.where(X > 0)
    arows = np.append(rows, i)
    acols = np.append(cols, j)
    urows = sdiff(els, arows)
    ucols = sdiff(els, acols)

    Xa = X.copy()
    Xa[i, j] = 1

    if len(rows) == n - 2:
        Xb = 0
        Xa[np.ix_(urows, ucols)] = 1
        lbb = np.trace(A @ Xa @ B @ Xa.T + 2 * C @ Xa.T)
        ubb = lbb
    else:
        Fa = A.copy()
        Da = B.copy()
        Ca = C.copy()

        # only keep unassigned rows, cols
        Fa = Fa[np.ix_(urows, urows)]
        Da = Da[np.ix_(ucols, ucols)]
        Ca = Ca[np.ix_(urows, ucols)]

        # new linear term
        Fi = A[np.ix_(arows, urows)]
        Dj = B[np.ix_(acols, ucols)]
        Ca = Ca + (Fi.T @ Dj)

        # compute constant term
        Fc = A[np.ix_(arows, arows)]
        Dc = B[np.ix_(acols, acols)]
        Cc = C[np.ix_(arows, acols)]
        cst = np.trace(Fc@Dc + Cc)

        try:
            lbb, ubb = admm_qap(Fa, Da, Ca, ub - cst, args)
            lbb = lbb + cst
            ubb = ubb + cst
        except Exception as e:
            print(f'Bound error: {e}')
            lbb = -float('inf')
            ubb = float('inf')
        #print(f'Assignment {i}->{j} | cst: {cst:.2f} | lb: {lbb:.2f} | ubb: {ub:.2f}')
    return lbb, ubb, Xa

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ex', type=str, default='nug12')
    parser.add_argument('--maxit', type=int, default=1000)
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--low_rank', action='store_true', default=False)
    parser.add_argument('--gamma', type=float, default=1.618)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    A = np.array([[1,1,1], [2,2,2], [3,3,3]])
    B = np.array([[1,1,1], [2,2,2], [3,3,3]])
    C = np.zeros(A.shape)
    X = np.zeros(A.shape)
    print('True sol: {}'.format(brute_solve(A, B, C)))
    print('Global lb, ub', admm_qap(A, B, C, args))
    print(bb_admm(A, B, C, 10, X, 1, 1, args))
