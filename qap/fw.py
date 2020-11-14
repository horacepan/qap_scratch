import pdb
import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy.io
from snpy.perm import Perm
from argparse import ArgumentParser

def qap_line_search(obj_func, A, B, X, X_new, tau=0.8):
    alpha = 1
    obj = obj_func(A, B, X)
    X_curr = X
    cnt = 0

    while obj_func(A, B, X_curr) >= obj:
        alpha = alpha * tau
        X_curr = (1 - alpha)*X + alpha*X_new
        cnt += 1

    return X_curr, alpha, cnt

def qap_grad(A, B, X):
    return B@X@A + B.T@X@A.T

def qap(A, B, X):
    return np.trace(A@X.T@B@X)

def perm_mat(rows, cols):
    n = len(rows)
    mat = np.zeros((n, n))
    for r, c in zip(rows, cols):
        mat[r, c] = 1.0

    return mat

def round_perm(perm):
    rows, cols = linear_sum_assignment(-perm)
    return perm_mat(rows, cols)

def fw_qap(A, B, args):
    tol = 1e-6
    maxit = args.maxit
    verbose = args.verbose

    n = A.shape[0]
    X = np.ones(A.shape) / n
    t = 0
    obj_prev = float('inf')

    for t in range(maxit + 1):
        obj = qap(A, B, X)

        if abs(obj - obj_prev) < tol:
            #break
            pass

        grad = qap_grad(A, B, X)
        opt_perm = perm_mat(*linear_sum_assignment(grad))

        if args.linesearch:
            X, alpha, cnt = qap_line_search(qap, A, B, X, opt_perm)
        else:
            alpha = 2 / (t + 2)
            X = (1 - alpha)*X + alpha*opt_perm
            cnt = 0

        t+= 1
        obj_prev = obj
        if verbose and t % 10 == 0:
            rounded_X = round_perm(X)
            rounded_obj = qap(A, B, rounded_X)
            print('Epoch {:4d} | Obj: {:4d} | Rounded obj: {:4d} | Cnt: {}'.format(t, int(obj), int(rounded_obj), cnt))

    opt = perm_mat(*linear_sum_assignment(-X))
    obj = qap(A, B, opt)
    print(f'Final value: {obj}')

def main(args):
    fname = f'../data/{args.ex}.mat'
    mats = scipy.io.loadmat(fname)
    A = mats['A']
    B = mats['B']
    fw_qap(A, B, args)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ex', type=str, default='nug12')
    parser.add_argument('--maxit', type=int, default=200)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--linesearch', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
