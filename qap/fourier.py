import time
import math
import numpy as np
import scipy.io

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


def cos_reps_minus2_pairs(n):
    reps_i = []
    reps_j = []
    for i in range(1, n+1):
        pi = Perm.cycle(i, n, n)
        reps_i.append(pi)

    for j in range(1, n):
        pj = Perm.cycle(j, n-1, n)
        reps_j.append(pj)
    return reps_i, reps_j

def get_fhat_cos(A, _lambda, cos_reps, cos_mats):
    n = sum(_lambda)
    d = hook_length(_lambda)
    mat = np.zeros((d, d))

    mat[-1, -1] = math.factorial(n - 2)
    tot = 0

    if _lambda == (n - 1, 1):
        mat[-2, -2] = mat[-1, -1]

    for c, crep in zip(cos_reps, cos_mats):
        tot += A[c(n) - 1, c(n-1) - 1] * (crep @ mat)

    return tot

def fhat_AB(A, B, _lambda, cos_reps, cos_mats):
    n = A.shape[0]
    fa = get_fhat_cos(A, _lambda, cos_reps, cos_mats)
    fb = get_fhat_cos(B, _lambda, cos_reps, cos_mats)
    return (fb@fa.T) / math.factorial(n - 2)

def get_fhats_sp(fname, tup):
    st = time.time()
    mat = scipy.io.loadmat(fname)
    A = mat['A']
    B = mat['B']
    n = A.shape[0]
    popt = Perm(tup)
    fhats = {}
    opt_reps = {}
    tot = 0
    print(f'Load time: {time.time()-st:.4f}s')

    # coset rep group elements
    reps_i, reps_j = cos_reps_minus2_pairs(n)
    reps = [i*j for i in reps_i for j in reps_j]

    for irrep in [(n-2, 1, 1), (n-2, 2), (n-1, 1), (n,)][::-1]:
        st = time.time()
        sp_rho = SnIrrep(irrep, 'sparse')
        mats_i = [sp_rho(i) for i in reps_i]
        mats_j = [sp_rho(j) for j in reps_j]
        rep_mats = [mi@mj for mi in mats_i for mj in mats_j]

        fh = fhat_AB(A, B, irrep, reps, rep_mats)
        rep = sp_rho(popt)
        tot += hook_length(irrep) * (rep.toarray()*fh).sum() / math.factorial(n)

        fhats[irrep] = fh
        opt_reps[irrep] = rep
        print(f'Time: {time.time() - st:.2f}s | Irrep: {irrep} | {type(fh)}')

    print(f'Total time: {time.time() - st:.2f}s')
    print(tot)
    return fhats, opt_reps

if __name__ == '__main__':
    fname = '../data/chr12a.mat'
    tup = (7,5,12,2,1,3,9,11,10,6,8,4)
    fhats, opt_rep = get_fhats_sp(fname, tup)
