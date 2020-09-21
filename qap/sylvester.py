import pdb
import time
import random
import os
from tqdm import tqdm
import numpy as np
from scipy.linalg import solve_sylvester # solve_sylvester(a, b, q) for AX + BX = Q
import matplotlib.pyplot as plt

from snpy.perm import Perm, sn
from snpy.sn_irrep import SnIrrep

def gen_block(rhos, g):
    n = sum([rho.dim for rho in rhos])
    block = np.zeros((n, n))

    idx = 0
    for rho in rhos:
        block[idx: idx+rho.dim, idx: idx+rho.dim] = rho(g)
        idx += rho.dim

    return block

def rand_perm(n, steps=100):
    gens = [Perm.trans(i, i+1, n) for i in range(1, n)]
    gens.append(Perm.cycle(1, n, n))
    p = Perm.eye(n)

    for i in range(steps):
        p = random.choice(gens) * p

    return p

def gen_intw(n):
    # try the intertwiner on Pkron X + X(-block) = 0
    rp = rand_perm(n)

    st = time.time()
    rhos = [SnIrrep((n - 2, 1, 1)),
            SnIrrep((n - 2, 2)),
            SnIrrep((n - 1, 1)), SnIrrep((n - 1, 1)), SnIrrep((n - 1, 1)),
            SnIrrep((n,)), SnIrrep((n,))]
    end = time.time()
    # print('Gen irreps: {:.2f}'.format(end - st))

    st = time.time()
    krp = np.kron(rp.mat(), rp.mat())
    end = time.time()
    # print('Kron time: {:.2f}'.format(end - st))

    blocked = gen_block(rhos, rp)
    C = np.eye(krp.shape[0]) * (1e-8)
    st = time.time()
    X = solve_sylvester(krp, -blocked, C)
    X = X / X.max()
    end = time.time()
    # print('Sylvester time: {:.2f}'.format(end - st))

    st = time.time()
    # Xinv = np.linalg.inv(X)
    end = time.time()
    # print('Inv time: {:.2f}'.format(end - st))

    # print(np.allclose(krp@X - X@blocked, 0, atol=1e-6))
    # print(np.allclose(Xinv@krp@X, blocked, atol=1e-6))
    # print('All zero?', np.allclose(X, 0, atol=1e-6))
    # print('X norm: {:.2f} | {:.2f}'.format(np.linalg.norm(X), np.linalg.norm(X, 'fro')))

    return X

def save_all(directory):
    for n in tqdm(range(12, 31)):
        mat = gen_intw(n)
        fname = os.path.join(directory, f'c{n}.npy')
        np.save(fname, mat)

if __name__ == '__main__':
    directory = '../intertwiners/'
    save_all(directory)
