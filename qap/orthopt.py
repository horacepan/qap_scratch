import numpy as np
from qap_utils import qap_func

def opt(func, grad_func, X, args):
    eta = args.eta
    tau = args.tau
    rho - args.rho
    maxit = args.maxit
    tol = args.tol

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

    for k in range(maxit):
        XP = X
        FP = F
        dtXP = dtX
        nls = 0
        deriv = rho * (normG**2)

        while True:
            X = np.linalg.solve(In + tau*H, XP - tau*RX)

            if np.linalg.norm(X.T@X - In, 'fro') > tol:
                X, _ = myQR(X)

            F = func(X)
            G = grad_func(X)

            if F <= C - tau * normG:
                break

            tau = eta * tau

        GX = G.T @ X
        GXT = G @ X.T
        H = 0.5 * (GXT - GXT.T)
        RX = H @ X
        dtX = G - X@GX

        S = X - XP
        Y = dtX - dtP
        if k % 2 == 0:
            tau = (S * S).sum() / SY
        else:
            tau = SY / np.abs(Y*Y).sum()

        QP = Q
        Q = gamma * Q + 1
        C = ((gamma * QP * C) + F) / Q

    return X

def main(args):
    n = 6
    A = np.random.random((n, n))
    B = np.random.random((n, n))
    X = np.random.random((n, n))

    qap_f, qap_g = qap_func(A, B)
    opt(qap_f, qap_g, X, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxit', type=int, default=1000)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--rho', type=float, default=1e-4)
    parser.add_argument('--eta', type=float, default=1e-1)
    parser.add_argument('--gamma', type=float, default=0.85)
    parser.add_argument('--tol', type=float, default=1e-5)
    args = parser.parse_args()
    main(args)
