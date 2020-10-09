import numpy as np

def qap_func(A, B, C=None):
    if C is None:
        C = np.zeros(A.shape)

    def fg(X):
        f = np.trace(A @ X @ B @ X.T) + 2 * np.trace(C.T @ X)
        return f

    def g(X):
        return (A + A.T) @ X @ (B + B.T) + (2 * C.T)

    return fg, g

def qap_func_hadamard(A, B, C=None):
    if C is None:
        C = np.zeros(A.shape)

    def fg(X):
        XX = X*X
        f = np.trace(A @ XX @ B @ XX.T) + 2 * np.trace(C.T @ XX)

    def g(X):
        XX = X * X
        grad = 2*(A @ XX @ B)*X + 2*(A.T @ XX @ B.T)*X + 4*C*X
        return grad

def myQR(mat):
    Q, R = np.linalg.qr(mat)
    rounded_r = np.sign(np.diag(R))
    if np.sum(rounded_r > 0) > 0:
        Q = Q * rounded_r
    return Q, R
