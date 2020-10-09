import pdb
import argparse
import numpy as np
from qap_utils import qap_func, myQR, qap_func_hadamard

def eig_prob(A):
    '''
    A is a symmetric matrix
    '''
    def f(X):
        return -0.5 * np.trace(X.T@A@X)

    def g(X):
        return -A@X

    return f, g

def opt(func, grad_func, X, args):
    eta = args.eta
    tau = args.tau
    rho = args.rho
    maxit = args.maxit
    tol = args.tol
    gamma = args.gamma

    n = X.shape[0]
    In = np.eye(n)
    F, G = func(X), grad_func(X)

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
            # X = np.linalg.solve(In + tau*H, XP - tau*H)

            if np.linalg.norm((X.T@X) - In, 'fro') > tol:
                X, _ = myQR(X)

            F = func(X)
            G = grad_func(X)

            if F <= C - tau * deriv + tol:
                break

            tau = eta * tau
            nls += 1
            print(f'Iter: {k:4d} | search iters: {nls:} | F: {F:.4f} | UB: {C - tau*deriv:.4f} | C: {C} | tau: {tau}')
            if nls > 30:
                pdb.set_trace()

        GX = G.T @ X
        GXT = G @ X.T
        H = 0.5 * (GXT - GXT.T)
        RX = H @ X
        dtX = G - X@GX

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
        print(f'Iter {k:4d} | f(X) = {F:.4f} | normG: {normG:.4f} | Xdiff: {Xdiff:.4f} | FDiff: {Fdiff:.4f}')
    return X

def main(args):
    n = 1000
    A = np.random.random((n, n))
    A = A + A.T
    B = np.random.random((n, n))
    X = np.random.random((n, n))
    X, _ = np.linalg.qr(X)

    Vs, _ = np.linalg.eig(A)
    print('Sum eigs: {}'.format(np.sum(Vs) * -0.5))

    f, g = eig_prob(A)
    Xopt = opt(f, g, X, args)
    print('Opt val: {:.2f}'.format(X))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxit', type=int, default=1000)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--rho', type=float, default=1e-1)
    parser.add_argument('--eta', type=float, default=1e-1)
    parser.add_argument('--gamma', type=float, default=0.85)
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
