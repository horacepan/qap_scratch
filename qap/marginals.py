import pdb
import numpy as np
import matplotlib.pyplot as plt

from snpy.perm import Perm, sn
from snpy.sn_irrep import SnIrrep
from snpy.utils import hook_length
from sylvester import rand_perm

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
    irreps = [(n - 2, 2), (n - 1, 1), (n - 1, 1), (n,)]
    cols = [-1, -2, -1, -1]
    return irreps, cols

def c_lambda(_lambda):
    '''
    Returns: the change of basis C such that
        C @ [perm module of young tabloid] = block diag @ C
    '''
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

    mat = np.concatenate(stacks, axis=0)
    if mat.shape[0] != mat.shape[1]:
        #print([hook_length(i) for i in irrep], irrep)
        #pdb.set_trace()
        pass
    return mat

def make_block_tensor(n, g):
    irreps = [(n - 2, 1, 1), (n - 2, 2), (n - 1, 1), (n - 1, 1), (n - 1, 1), (n,), (n,)]
    hls = [hook_length(ir) for ir in irreps]
    tot = int(sum(hls))
    mat = np.zeros((tot, tot))

    idx = 0
    for ir, d in zip(irreps, hls):
        d = int(d)
        rho = SnIrrep(ir)
        mat[idx: idx+d, idx: idx+d] = rho(g)
        idx += d

    return mat

if __name__ == '__main__':
    n = 10
    g = rand_perm(10)
    mat = make_block_tensor(n, g)
    plt.spy(np.round(mat, 4))
    plt.show()
