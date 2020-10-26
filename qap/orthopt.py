import time
import os
import pdb
from copy import deepcopy
import argparse
import numpy as np
import scipy.io
from qap_utils import qap_func, myQR, qap_func_hadamard, qap_func_hadamard_lagrangian, isperm
from multiprocessing import Pool

def eig_prob(A):
    '''
    A is a symmetric matrix
    '''
    def f(X, a1=None, a2=None):
        return -0.5 * np.trace(X.T@A@X)

    def g(X, a1=None, a2=None):
        return -A@X

    return f, g

def opt(func, grad_func, X, mu, _lambda, args):
    eta = args.eta
    tau = args.tau
    rho = args.rho
    maxit = args.maxit
    tol = args.tol
    gamma = args.gamma

    n, p = X.shape
    In = np.eye(n)
    Ip = np.eye(p)
    _lambda = np.zeros(X.shape)
    F, G = func(X, _lambda, mu), grad_func(X, _lambda, mu)

    GX = G.T @ X
    GXT = G @ X.T
    dtX = G - X@GX
    H = 0.5 * (GXT - GXT.T)
    RX = H @ X

    normG = np.linalg.norm(dtX, 'fro')
    C = F
    Q = 1
    X0 = X.copy()

    for k in range(maxit):
        XP = X
        FP = F
        dtXP = dtX
        nls = 0
        deriv = rho * (normG**2)

        while True:
            X = np.linalg.solve(In + tau*H, XP - tau*RX)

            if np.linalg.norm((X.T@X) - Ip, 'fro') > tol:
                X, _ = myQR(X)

            F = func(X, _lambda, mu)
            G = grad_func(X, _lambda, mu)
            if F <= C - tau*deriv:
                break

            tau = eta * tau
            nls += 1
            if nls > 5:
                break

        GX = G.T @ X # note this is the lagrange multiplier too
        GXT = G @ X.T
        H = 0.5 * (GXT - GXT.T)
        RX = H @ X
        dtX = G - X@GX
        normG = np.linalg.norm(dtX, 'fro')

        S = X - XP
        Y = dtX - dtXP
        SY = (S*Y).sum()
        if k % 2 == 0:
            tau = (S * S).sum() / SY
        else:
            tau = SY / np.abs(Y*Y).sum()

        QP = Q
        Q = gamma * Q + 1
        C = ((gamma * QP * C) + F) / Q

        Xdiff = np.linalg.norm(X - XP, 'fro')
        Fdiff = np.abs(F - FP) / (abs(FP) + 1)
        if normG < tol or Xdiff < tol:
            if args.verbose:
                print(f'Breaking at iter {k} | normG: {normG:.8f} | Xdiff: {Xdiff:.8f} | Fdiff: {Fdiff:.8f}')
            break

        if k % 10 == 0 and args.verbose:
            print(f'Iter {k:4d} | f(X) = {F:.4f} | normG: {normG:.4f} | Xdiff: {Xdiff:.4f} | FDiff: {Fdiff:.4f}')
    return X

def main(args):
    n = 100
    p = 6
    A = np.random.random((n, n))
    A = A + A.T
    X = np.random.random((n, p))
    X, _ = np.linalg.qr(X)

    Vs, Xs = np.linalg.eig(A)
    true_eig = sorted(Vs)[-p:]
    f, g = eig_prob(A)
    mu = None
    _lambda = None

    print('Opt eigs eval: {:.4f}'.format(-sum(true_eig) * 0.5))
    Xopt = opt(f, g, X, mu, _lambda, args)
    print('Opt val: {:.4f}'.format(f(Xopt)))

def qap_main(args):
    np.random.seed(args.seed)
    mats= scipy.io.loadmat(args.ex)
    A = mats['A']
    B = mats['B']

    f, g = qap_func_hadamard_lagrangian(A, B)

    f_true, g_true = qap_func_hadamard(A, B)
    _lambda = np.random.random(A.shape)

    X = np.random.random(A.shape)
    X, _ = np.linalg.qr(X)
    mu = args.mu
    prev = float('inf')

    for i in range(101):
        X = opt(f, g, X, mu, _lambda, args)
        if i%10 == 0:
            curr = f(X, _lambda, mu)
            if args.verbose:
                print(f'Iter {i:3d} | f(X) = {f(X, _lambda, mu):.3f} | isperm: {isperm(X)}')
            if np.allclose(prev, curr):
                break
            prev = curr

        _lambda = np.max(_lambda - mu*X, 0)
        mu = 1.2 * mu

    Xround = np.round(X)
    if args.verbose:
        print('Rounded f(X): {:.3f}'.format(f(Xround, _lambda, mu)))
        print('Rounded nopenalty f(X): {:.3f} | is perm: {}'.format(f_true(Xround), isperm(Xround)))
    return f_true(Xround)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ex', type=str, default='had12')
    parser.add_argument('--maxit', type=int, default=100)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--rho', type=float, default=1e-4)
    parser.add_argument('--eta', type=float, default=1e-1)
    parser.add_argument('--gamma', type=float, default=0.85)
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    os.chdir('../data/')
    all_args = []
    for fname in os.listdir():
        if '.mat' in fname:
            args.ex = fname
        if 'nug12.mat' not in fname:
            continue
        print('File: {}'.format(args.ex))
        opt_res = float('inf')
        results = []

        _st = time.time()
        for seed in range(40):
            #for eta in  [0.01, 0.1, 1]:
            for eta in  [0.5]:
                args.eta = eta
                #for gamma in [0.1, 0.5, 0.85, 1]:
                for gamma in [0.8]:
                    args.gamma = gamma
                    for mu in [5]:
                        st = time.time()
                        args.seed = seed
                        args.mu = mu
                        args_copy = deepcopy(args)
                        all_args.append(args_copy)
                        '''
                        all_args.append(args_copy)
                        res = qap_main(args)
                        el = time.time() - st
                        opt_res = min(opt_res, res)
                        print(f'Seed: {seed:2d} | eta: {str(eta):4s} | gamma: {str(gamma):4s} | mu: {str(mu):4s} | res: {res:.2f} | time: {el:.2f}s')
                        results.append(res)
                        '''
        #print(f'Opt res: {opt_res:.2f} | Median: {np.median(results)} | Mean: {np.mean(results):.2f} | Total time: {time.time() - _st:.2f}s')
        print('Starting pool')
        npool = 4
        with Pool(npool) as p:
            st = time.time()
            results = p.map(qap_main, all_args, chunksize=len(all_args)//npool)
            end = time.time()
            print(np.min(results), np.median(results), np.mean(results))
            print('Elapsed:', end - st)
