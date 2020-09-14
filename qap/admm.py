import pdb
import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

PREFIX = '../data/datasets1/'

def make_vhat(V):
    n = V.shape[0]
    vxv = np.kron(V, V)
    r1 = np.zeros((1, vxv.shape[-1] + 1))
    r1[0] = np.sqrt(0.5)

    r2 = np.concatenate([np.ones((n * n, 1)) * (np.sqrt(2) / n), vxv], axis=1)
    Vhat = np.concatenate([r1, r2])
    return Vhat

def make_gangster(n):
    J = np.zeros((n*n + 1, n*n + 1))
    eye_n = np.eye(n)
    diag_zero = np.ones((n, n))
    np.fill_diagonal(diag_zero, 0)

    J[0, 0] = 1

    for i in range(n):
        for j in range(n):
            if i == j:
                J[1+i*n: 1+(i+1)*n, 1+j*n: 1+(j+1)*n] = diag_zero
            else:
                J[1+i*n: 1+(i+1)*n, 1+j*n: 1+(j+1)*n] = eye_n
    return J.astype(bool)

def admm_qap(L, Vhat, J, args):
    st = time.time()
    maxit = args.maxit
    tol = args.tol
    beta = args.beta
    gamma = args.gamma
    low_rank = args.low_rank
    K = args.K

    normL = np.linalg.norm(L)
    Vhat_nrows = Vhat.shape[0]
    L = L / (normL * Vhat_nrows)
    n = int((L.shape[0] - 1) ** 0.5)
    r_shape = None
    Y = np.random.random((Vhat.shape[0], Vhat.shape[0]))
    Yhat = np.random.random((Vhat.shape[0], Vhat.shape[0]))
    R = Vhat.T @ Yhat @ Vhat
    Z = Y - (Vhat @ R @  Vhat.T)

    for i in tqdm(range(maxit)):
        R_pre_proj = Vhat.T @ (Y + Z / beta) @ Vhat
        R_pre_proj = (R_pre_proj + R_pre_proj.T) / 2.

        S, U = np.linalg.eig(R_pre_proj)
        if args.low_rank:
            if S[-1] > 0:
                vhat_u = Vhat @ U[:, -1:]
                VRV = S[-1] * vhat_u @ vhat_u.T
                #R = S[-1] * U[:, -1:] @ U[:, -1:].T
            else:
                VRV = np.zeros(U.shape)
        else:
            pos_idx = S > 0
            vhat_u = Vhat @ U[:, pos_idx]
            VRV = (vhat_u * S[pos_idx]) @ vhat_u.T
            #R = U[:, pos_idx] @ np.diag(S[pos_idx]) @ U[:, pos_idx].T

        # update Y
        Y = VRV - (L + Z) / beta
        Y[J] = 0
        Y[0, 0] = 1
        Y = np.minimum(1, np.maximum(0, Y))
        Y[0, 0] = 1
        Y[J] = 0
        Y[np.abs(Y) < tol] = 0

        # update Z
        Z = Z + gamma * beta * (Y - VRV)
        Z = (Z + Z.T) / 2.
        Z[np.abs(Z) < tol] = 0
        # the computed lower bound is ...

    tt = time.time() - st
    print('Done! | Elapsed: {:.2f}min'.format(tt / 60))
    print(lower_bound(L, J, Vhat, Z, n, scale=normL * Vhat_nrows))

def lower_bound(L, J, Vhat, Z, n, scale=1):
    In = np.eye(n)
    en = np.ones((n, 1))
    kron_ier = np.kron(In, en.T)
    kron_eri = np.kron(en.T, In)
    krons = np.concatenate([kron_ier, kron_eri])
    That = np.concatenate([-np.ones((2*n, 1)), krons], 1)

    # Get the nullspace of Vhat?
    Q, _ = np.linalg.qr(That.T, mode='reduced')
    Uloc = np.concatenate([Vhat, Q], 1)
    Zloc = Uloc.T @ Z @ Uloc
    W12 = Zloc[:(n-1)*(n-1) + 1, (n-1)*(n-1):]
    W21 = Zloc[(n-1)*(n-1)+1:, :(n-1)*(n-1)+1]
    W22 = Zloc[(n-1)*(n-1)+1:, (n-1)*(n-1)+1:]
    W11 = Zloc[:(n-1)*(n-1)+1, :(n-1)*(n-1)+1]
    W11 = (W11 + W11.T) / 2.

    # Project W11 onto negative definite matrices for Zp
    Dw, Uw = np.linalg.eig(W11)
    neg_idx = Dw < 0
    pdb.set_trace()
    W11 = (Uw[:, neg_idx] * Dw[neg_idx]) @ Uw[:, neg_idx].T
    Zp = Uloc @ np.block([[W11, W12], [W21, W22]]) @ Uloc.T
    Zp = (Zp + Zp.T) / 2

    # dont know why we dont just use Y_out
    Yp = np.zeros(L.shape)
    Yp[L + Zp < 0] = 1
    Yp[J] = 0
    Yp[0, 0] = 1
    return ((L + Zp) * Yp).sum() * scale

def main(args):
    np.random.seed(args.seed)
    fname = os.path.join(PREFIX, args.ex + '.mat')
    mats = scipy.io.loadmat(fname)
    A = mats['A']
    B = mats['B']
    n = A.shape[0]

    Ldim = 1 + A.shape[0] * A.shape[0]
    L = np.zeros((Ldim, Ldim))
    L[1:, 1:] = np.kron(B, A)
    V = np.concatenate([np.eye(n - 1), np.ones((1, n - 1))])
    Vhat = make_vhat(V)
    J = make_gangster(n)
    pdb.set_trace()
    admm_qap(L, Vhat, J, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ex', type=str, default='nug12')
    parser.add_argument('--maxit', type=int, default=1000)
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--low_rank', action='store_true', default=False)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--gamma', type=float, default=1.618)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
