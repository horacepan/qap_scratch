import numpy as np

from snpy.perm import Perm, sn
from snpy.sn_irrep import SnIrrep

def coset_reps_2(n):
    reps = []
    for i in range(1, n + 1):
        pi = Perm.cycle(i, n, n)
        for j in range(1, n):
            pj = Perm.cycle(j, n - 1, n)
            pij = pi * pj
            reps.append(pij)

    return reps

def coset_reps_1(n):
    reps = [Perm.cycle(i, n, n) for i in range(1, n + 1)]
    return reps

def qap_decompose(_lambda):
    n = sum(_lambda)
    if _lambda == (n - 1, 1):
        return decompose_n11(_lambda)
    elif _lambda == (n - 2, 2):
        return decompose_n22(_lambda)
    elif _lambda == (n - 2, 1, 1):
        return decompose_n211(_lambda)
    else:
        print(f'Cant decompose this for the QAP: {_lambda}')
        return

def decompose_n11(_lambda):
    n = sum(_lambda)
    irreps = [(n - 1, 1), (n,)]
    cols = [-1, -1]
    return irreps, cols

def decompose_n211(_lambda):
    n = sum(_lambda)
    irreps = [(n - 2, 1, 1), (n - 2, 2), (n - 1, 1), (n - 1, 1), (n,)]
    cols = [-1, -1, -2, -1, -1]
    return irreps, cols

def decompose_n22(_lambda):
    n = sum(_lambda)
    irreps = [(n - 2, 2), (n - 1, 1), (n,)]
    cols = [-1, -1, -1]
    return irreps, cols

def c_lambda(_lambda):
    n = sum(_lambda)
    irrep, cols = qap_decompose(_lambda)
    if _lambda[0] == n - 1:
        cos_reps = coset_reps_1(n)
    elif _lambda[0] == n - 2:
        cos_reps = coset_reps_2(n)
    else:
        raise Exception('Not implemented')

    stacks = []
    for part, c in zip(irrep, cols):
        rho = SnIrrep(part)
        _cols = []
        # loop over coset representatives
        for p in cos_reps:
            column = rho(p)[:, c]
            _cols.append(column)
        pstack = np.stack(_cols).T
        stacks.append(pstack)

    return np.concatenate(stacks, axis=0)
