import pdb
import argparse
import numpy as np
import scipy.io
from qap_utils import qap_func, myQR, qap_func_hadamard

def eig_prob(A):
    '''
    A is a symmetric matrix
    '''
    def f(X, a1=None, a2=None):
        return -0.5 * np.trace(X.T@A@X)

    def g(X, a1=None, a2=None):
        return -A@X

    return f, g

def opt(func, grad_func, X, mu, args):
    eta = args.eta
    tau = args.tau
    rho = args.rho
    maxit = args.maxit
    tol = args.tol
    gamma = args.gamma

    n, p = X.shape
    _lambda = np.zeros(X.shape)
    In = np.eye(n)
    Ip = np.eye(p)
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
                print(f'Breaking at iter {k} | normG: {normG:.2f} | Xdiff: {Xdiff:.2f} | Fdiff: {Fdiff:.2f}')
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

    print('Opt eigs eval: {:.4f}'.format(-sum(true_eig) * 0.5))
    Xopt = opt(f, g, X, mu, args)
    print('Opt val: {:.4f}'.format(f(Xopt)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxit', type=int, default=10000)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--rho', type=float, default=1e-4)
    parser.add_argument('--eta', type=float, default=1e-1)
    parser.add_argument('--gamma', type=float, default=0.85)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
