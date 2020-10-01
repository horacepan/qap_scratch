import numpy as np

def qap_func(A, B, C=None):
    if C is None:
        C = np.zeros(A.shape)

    def fg(X):
        f = np.trace(A @ X @ B @ X.T) + 2 * np.trace(C.T @ X)
        return f

    def g(X):
        return (A + A.T) @ X @ (B + B.T) + (2 * C.T)

    return f, g

def opt(func, grad_func, X, rho, delta, eta, eps, maxit):
    k = 0
    n = X.shape[0]
    f, g0 = func(X), grad_func(X)
    In = np.eye(n)
    C = f
    Q = 1

    for k in range(maxit):
        while True:
            Yk_tau = np.solve(In + tau/2, In - tau/2)
            fyk_tau = func(Yk_tau)

            if fyk_tau <= C + rho1 * tau * g0:
                break

            tau = delta * tau

        Xprev = X
        grad_prev = grad_func(X)
        X = Yk_tau
        grad_now = grad_func(X)

        Q = eta * Q + 1
        C = ((eta * Q * C) + f(X)) / Q
        S = X - Xprev
        Yk = grad_now - grad_prev

        # update tau
        SY = (S*Y).sum()
        tau_k1 = (S * S).sum() / SY
        tau_k2 = SY / np.abs(Y*Y).sum()

    return X

def main():
    A = np.random.random((n, n))
    B = np.random.random((n, n))
    X = np.random.random((n, n))

    qap_f, qap_g = qap_func(A, B)
    rho = 1e-2
    nabla = 1e-2
    eta = 1e-2
    eps = 1e-2
    maxit = 1000
    opt(qap_f, qap_g, X, rho, nabla, eta, eps, maxit)

if __name__ == '__main__':
    main()
