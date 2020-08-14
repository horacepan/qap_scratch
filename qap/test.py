import pdb
import math
import numpy as np

from snpy.perm import Perm, sn
from snpy.sn_irrep import SnIrrep
from snpy.utils import hook_length

from marginals import c_lambda, qap_decompose

def sn_distr(n):
    distr = np.random.random(math.factorial(n))
    distr = np.exp(distr)
    distr = distr / np.sum(distr)
    return distr

def make_sn_func(distr, _sn):
    p_to_idx = perm_to_idx(_sn)
    def f(p):
        idx = p_to_idx[p.tup]
        return distr[idx]
    return f

def perm_to_idx(_sn):
    indices = {}

    for idx, p in enumerate(_sn):
        indices[p.tup] = idx

    return indices

def perm_rep(g):
    return g.mat()

def ordered_tups(n):
    all_tups = []

    for i in range(1, n+1):
        for j in range(1, n+1):
            if i == j:
                continue
            tup = (i, j)
            all_tups.append(tup)

    return all_tups

def perm_tup(p, all_tups=None):
    if all_tups is None:
        all_tups = ordered_tups(p.size)

    mat = np.zeros((len(all_tups), len(all_tups)))
    for idx, tup in enumerate(all_tups):
        # p(j) = i
        ptup = tuple(p[i] for i in tup)
        j = all_tups.index(ptup)

        mat[j, idx] = 1

    return mat

def ft(f, rho, group):
    fhat = 0

    for g in group:
        fhat += f(g) * rho(g)

    return fhat

def gen_fhat_blocks(f, _lambdas, group):
    d = int(sum([hook_length(l) for l in _lambdas]))
    fhat = np.zeros((d, d))

    idx = 0
    for irrep in _lambdas:
        rho = SnIrrep(irrep)
        mat = ft(f, rho, group)
        k = mat.shape[0]
        fhat[idx: idx + k, idx: idx + k] = mat
        idx += k

    return fhat

def main():
    n = 6
    _sn = sn(n)
    _lambda = (n - 2, 1, 1)

    distr = sn_distr(n)
    sn_func = make_sn_func(distr, _sn)
    irreps, _ = qap_decompose(_lambda)

    all_tups = ordered_tups(n)
    fhat_perm = ft(sn_func, perm_rep, _sn)
    fhat_perm_tup = ft(sn_func, lambda p: perm_tup(p, all_tups), _sn)
    block_diag = gen_fhat_blocks(sn_func, irreps, _sn)
    c_mat = c_lambda(_lambda)
    print(np.allclose(np.linalg.inv(c_mat) @ block_diag @ c_mat, fhat_perm_tup))

if __name__ == '__main__':
    main()
