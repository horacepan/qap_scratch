import numpy as np
from scipy.optimize import linear_sum_assignment

def glb(A, B):
    '''
    A: n x n numpy matrix
    B: n x n numpy matrix
    Returns: the Gilmore Lawler lower bound
    '''
    n = A.shape[0]
    C = np.kron(A, B)
    l_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A_srt = np.delete(A[i, :], i, 0)
            B_srt = np.delete(B[j, :], j, 0)
            A_srt.sort()
            B_srt.sort()
            l_mat[i, j] = A[i, i] + B[j, j] + A_srt.dot(B_srt[::-1])

    lrows, lcols = linear_sum_assignment(l_mat)
    total = l_mat[lrows, lcols].sum()
    return total

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
        return f

    def g(X):
        XX = X * X
        grad = 2*(A @ XX @ B)*X + 2*(A.T @ XX @ B.T)*X + 4*C*X
        return grad

    return fg, g

def qap_func_hadamard_lagrangian(A, B, C=None):
    qapf, qapg = qap_func_hadamard(A, B, C)

    def f(X, _lambda, mu):
        mask = mu*X < _lambda
        penalty = ((-_lambda * X) + 0.5*mu*(X*X)) * mask
        penalty = penalty + ((-_lambda * _lambda) / (2 * mu)) * (1 - mask)
        tot = qapf(X) + np.sum(penalty)
        return tot

    def g(X, _lambda, mu):
        mask = mu*X < _lambda
        penalty = (-_lambda + (mu * X)) * mask
        return qapg(X) + penalty

    return f, g

def myQR(mat):
    Q, R = np.linalg.qr(mat)
    rounded_r = np.sign(np.diag(R))
    if np.sum(rounded_r > 0) > 0:
        Q = Q * rounded_r

    return Q, R

def isperm(X, tol=1e-8):
    n = X.shape[0]
    ones = np.ones(n)
    return np.allclose(X.dot(ones), 1) and np.allclose(X.T.dot(ones), 1) and np.all(X > -tol)
