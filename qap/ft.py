import time
import pdb
from snpy.sn_irrep import SnIrrep
from snpy.perm import Perm, sn
import scipy.io
import numpy as np
import math


def sn_minus2_coset(n):
    reps = []

    for i in range(1, n+1):
        pi = Perm.cycle(i, n, n)
        for j in range(1, n):
            pj = Perm.cycle(j, n-1, n)
            reps.append(pi * pj)

    return reps

def raw_ft(f, _lambda):
    n = sum(_lambda)
    _sn = sn(n)
    rho = SnIrrep(_lambda, fmt='dense')
    fhat = 0

    for p in _sn:
        fhat += rho(p) * f(p)

    return fhat

def qap_ft(mat, _lambda):
    '''
    We will only ever take the FT of: (n-1, 1), (n-2, 2), (n-2, 1, 1)
    '''
    n = sum(_lambda)
    rho = SnIrrep(_lambda, fmt='dense')
    fhat = 0
    cos_reps = sn_minus2_coset(n)
    sn_minus2_size = math.factorial(n) / len(cos_reps)
    fmat = np.zeros(rho(Perm.eye(n)).shape)
    fmat[-1, -1] = sn_minus2_size

    if _lambda == (n - 1, 1):
        fmat[-2, -2] = sn_minus2_size

    for p in cos_reps:
        feval = mat[p(n) - 1, p(n-1) - 1]
        fhat += feval * (rho(p) @ fmat)

    return fhat

def gen_qap_func(a, b):
    def f(p):
        pmat = p.mat()
        pb = (pmat @ b) @ pmat.T
        return (a * pb).sum()
    return f

if __name__ == '__main__':
    _lambda = (4, 2)
    amat = np.random.randint(0, 10, size=(6, 6))
    bmat = np.random.randint(0, 10, size=(6, 6))

    st = time.time()
    ahat = qap_ft(amat, _lambda)
    bhat = qap_ft(bmat, _lambda)
    fhat = (1 / math.factorial(4)) * (ahat @ bhat.T)
    print('Elapsed: {:.2f}s'.format(time.time() - st))

    st = time.time()
    fab = gen_qap_func(amat, bmat)
    fhat2 = raw_ft(fab, _lambda)
    print(np.allclose(fhat2, fhat))
    print('Elapsed: {:.2f}s'.format(time.time() - st))
    pdb.set_trace()
