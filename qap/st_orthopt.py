import pdb
import numpy as np
from qap_utils import qap_func, myQR

def opt(func, grad_func, X, rho, eta, tau, rho1, eps, maxit):
    k = 0
    n = X.shape[0]
    f, G = func(X), grad_func(X)
    In = np.eye(n)
    C = f
    Q = 1

    GXt = G@X.T
    A = GXt - GXt.T
    dtX = G - X@G.T@X
    normG = np.linalg.norm(dtX, 'fro')
    deriv = rho1 * (normG**2)
    A = G - X@G.T@X
    grad_now = G
    for k in range(maxit):
        nls = 0
        deriv = rho1 * (normG**2)
        while True:
            htauA = (tau/2) * A
            Yk_tau = np.linalg.solve(In + htauA, In - htauA)

            eye_dist = np.linalg.norm(X.T@X - In)
            if eye_dist > 1e-6:
                _Q, _R = myQR(X)
                X = _Q
                f = func(X)

            if nls > 0:
                print(f'Line search: {nls:4d} | f: {f:.2f} | C - deriv: {C - deriv:.2f}')

            if f <= (C - tau * deriv):
                break
            tau = eta * tau

            nls += 1
        print(f' num nls: {nls}')

        A = GXt - GXt.T
        Xprev = X
        grad_prev = grad_now # is this where it shouldbe?
        X = Yk_tau
        grad_now = grad_func(X)
        dtX = G - X@G.T@X
        normG = np.linalg.norm(dtX, 'fro')

        Q = eta * Q + 1
        C = ((eta * Q * C) + f) / Q
        S = X - Xprev
        Yk = grad_now - grad_prev
        Y = Yk

        # update tau
        SY = (S*Y).sum()
        if k % 2 == 0:
            tau = tau_k1 = (S * S).sum() / SY
        else:
            tau = tau_k2 = SY / np.abs(Y*Y).sum()

        tau = max(min(tau, 1e20), 1e-20)
        dtX = G - X@G.T@X
        normG = np.linalg.norm(dtX, 'fro')

        if k % 10 == 0:
            print(f'Iter {k} | G norm: {normG:.2f}')
    return X

def main():
    n = 6
    A = np.random.random((n, n))
    B = np.random.random((n, n))
    X = np.random.random((n, n))

    qap_f, qap_g = qap_func(A, B)
    rho = 1e-2
    eta = 1e-1
    eps = 1e-2
    tau = 1e-3
    rho1 = 1e-4
    maxit = 1000
    opt(qap_f, qap_g, X, rho, eta, tau, rho1, eps, maxit)

if __name__ == '__main__':
    main()
