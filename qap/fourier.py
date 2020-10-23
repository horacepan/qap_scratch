import numpy as np

from snpy.perm import Perm, sn
from snpy.utils import hook_length
from snpy.sn_irrep import SnIrrep

def cos_reps_minus2(n):
    '''
    n: int
    Returns: list of coset representatives for S_n / S_{n-2}
    '''
    reps = []
    for i in range(1, n+1):
        pi = Perm.cycle(i, n, n)
        for j in range(1, n):
            pj = Perm.cycle(j, n-1, n)
            reps.append(pi * pj)
    return reps

def fhat_mat(A, _lambda, cos_reps=None):
    '''
    A: numpy matrix of size n x n
    _lambda: partition of n
    cos_reps: (optional) list of coset representatives for S_n / S_{n-2}
    Returns: fourier matrix of the graph function given by A at irrep _lambda
    '''
    n = sum(_lambda)
    rho = SnIrrep(_lambda)
    d = hook_length(_lambda)
    mat = np.zeros((d, d))
    mat[-1, -1] = math.factorial(n - 2)
    tot = 0

    if _lambda == (n - 1, 1):
        mat[-2, -2] = mat[-1, -1]

    if cos_reps is None:
        cos_reps = cos_reps_minus2(n)

    for c in cos_reps:
        tot += A[c(n) - 1, c(n-1) - 1] * (rho(c) @ mat)

    return tot

def fhat_qap(A, B, _lambda):
    '''
    A: numpy matrix of size n x n
    B: numpy matrix of size n x n
    _lambda: partition of n
    Returns: the fourier matrix for irrep _lambda
    '''
    n = A.shape[0]
    cos_reps = cos_reps_minus2(n)
    fa = fhat_fast_mat(A, _lambda, cos_reps)
    fb = fhat_fast_mat(B, _lambda, cos_reps)
    return (fa@fb.T) / math.factorial(n - 2)
