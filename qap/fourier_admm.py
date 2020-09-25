import pdb
import os
import time
import random
from tqdm import tqdm
import argparse
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from admm import make_vhat, make_gangster, make_L, make_y0, make_r0, lower_bound
from snpy.utils import hook_length

PREFIX = '../data/datasets1/'

def load_cg_transform(n, directory):
    fname = os.path.join(directory, f'c{n}.npy')
    return np.load(fname)

def make_blocked(blocks, n):
    bmat = np.zeros((n * n, n * n))
    irreps = [(n - 2, 1, 1), (n - 2, 2), (n - 1, 1), (n - 1, 1), (n - 1, 1), (n,), (n,)]
    idx = 0

    for irrep in irreps:
        dim = hook_length(irrep)
        bmat[idx: idx+d, idx: idx+d] = irreps[irrep]
        idx += dim

    return bmat

def fourier_admm_qap(L, Vhat, J, args, n, outer_AB):
    st = time.time()
    maxit = args.maxit
    tol = args.tol
    gamma = args.gamma
    low_rank = args.low_rank
    K = args.K

    normL = np.linalg.norm(L)
    Vhat_nrows = Vhat.shape[0]
    L = L * (n*n / normL)
    mask = make_bdiag_mask(n)
    beta = n / 3.

    Y0 = make_y0(n)
    # R0 = make_r0(n, Vhat, Y0)
    R0 = np.eye((n-1)**2 + 1)
    Z0 = Y0 - (Vhat @ R0 @ Vhat.T)

    Y = Y0
    R = R0
    Z = Z0
    C = load_cg_transform(n, '../intertwiners/c{n}.npy')
    blocks = {
        (n - 2, 1, 1): np.eye(hook_length((n - 2, 1, 1))),
        (n - 2, 2)   : np.eye(hook_length((n - 2, 2))),
        (n - 1, 1)   : np.eye(hook_length((n - 1, 1))),
        (n,)         : np.eye(1),
    }
    B = make_blocked(blocks, n)

    for i in tqdm(range(maxit)):
        R_pre_proj = Vhat.T @ (Y + Z / beta) @ Vhat
        R_pre_proj = (R_pre_proj + R_pre_proj.T) / 2.

        S, U = np.linalg.eigh(R_pre_proj)
        if not args.low_rank:
            pos_idx = S > 0
            if pos_idx.sum() > 0:
                vhat_u = Vhat @ U[:, pos_idx]
                VRV = (vhat_u * S[pos_idx]) @ vhat_u.T
            else:
                VRV = np.zeros(Y.shape)
        else:
            if S[-1] > 0:
                vhat_u = Vhat @ U[:, -1:]
                VRV = S[-1] * vhat_u @ vhat_u.T
            else:
                VRV = np.zeros(Y.shape)

        # update Y
        # Y = VRV - ((L + Z) / beta)
        Y = CBC - ((L + Z) / beta)
        Y = (Y + Y.T) / 2.
        Y[J] = 0; Y[0,0] = 1
        Y = np.minimum(1, np.maximum(0, Y))
        Y[J] = 0; Y[0,0] = 1
        Y[np.abs(Y) < tol] = 0
        pR = Y - VRV

        # update Z
        # Z = Z + gamma * beta * (Y - VRV)
        Z = Z + gamma * beta * (Y - CBC)
        Z = (Z + Z.T) / 2.
        Z[np.abs(Z) < tol] = 0

        if i % 100 == 0:
            scale = normL / (n * n)
            npr = np.linalg.norm(pR, 'fro')
            lbd = lower_bound(L, J, Vhat, Z, n, scale=scale)
            dy = np.square(Y.dot(np.ones(len(Y))) - np.ones(len(Y))).sum()
            dyt = np.square(Y.T.dot(np.ones(len(Y))) - np.ones(len(Y))).sum()
            print(f'Iter {i} | Lower bound: {lbd:.2f} | pR: {npr:.6f} | Ye - e: {dy:.4f} {dyt:.4f}')

    tt = time.time() - st
    print('Done! | Elapsed: {:.2f}min'.format(tt / 60))

def main(args):
    np.random.seed(args.seed)
    fname = os.path.join(PREFIX, args.ex + '.mat')

    mats = scipy.io.loadmat(fname)
    A = mats['A']
    B = mats['B']
    n = A.shape[0]

    # cg = load_cg_transform(n, directory=args.directory)
    outer_AB = np.reshape(A, (-1, 1), 'F') @ np.reshape(B, (-1, 1), 'F').T
    Vhat = make_vhat(n)
    J = make_gangster(n)
    L = make_L(A, B)
    fourier_admm_qap(L, Vhat, J, args, n, outer_AB)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ex', type=str, default='nug12')
    parser.add_argument('--maxit', type=int, default=1000)
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--low_rank', action='store_true', default=False)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=1.618)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    main(args)
