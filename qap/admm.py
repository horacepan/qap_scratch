import pdb
import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

PREFIX = '../data/datasets1/'

def psd_project(mat):
    S, U = np.linalg.eig(mat)
    pos_idx = S > 0
    return (U[:, pos_idx] * S[pos_idx]) @ U[:, pos_idx].T

def make_y0(n):
    n2 = n * n
    yhat = np.zeros((n2 + 1, n2 + 1))
    nI_e = n * np.eye(n) - 1

    yhat[0, 0] = 1
    yhat[0, 1:] = 1 / n
    yhat[1:, 0] = 1 / n
    yhat[1:, 1:] = (1 / n2) + ((1 / (n2 * (n - 1))) * np.kron(nI_e, nI_e))
    return yhat

def make_r0(n, Vhat, Yhat):
    R0 = Vhat.T @ Yhat @ Vhat
    R0 = (R0 + R0.T) / 2.
    return R0

def make_vhat(V):
    n = V.shape[0]
    vxv = np.kron(V, V)
    r1 = np.zeros((1, vxv.shape[-1] + 1))
    r1[0] = np.sqrt(0.5)

    r2 = np.concatenate([np.ones((n * n, 1)) * (np.sqrt(2) / n), vxv], axis=1)
    Vhat = np.concatenate([r1, r2])
    return Vhat

def make_that(n):
    In = np.eye(n)
    en = np.ones((n, 1))
    kron_ier = np.kron(In, en.T)
    kron_eri = np.kron(en.T, In)
    krons = np.concatenate([kron_ier, kron_eri])
    That = np.concatenate([-np.ones((2*n, 1)), krons], 1)
    return That

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
    gamma = args.gamma
    low_rank = args.low_rank
    K = args.K


    normL = np.linalg.norm(L)
    Vhat_nrows = Vhat.shape[0]
    L = L / normL
    n = int((L.shape[0] - 1) ** 0.5)
    beta = n / 3.

    Y0 = make_y0(n)
    R0 = make_r0(n, Vhat, Y0)
    Z0 = Y0 - (Vhat @ R0 @ Vhat.T)
    print('Y0', np.linalg.norm(Y0, 'fro'))
    print('Y, R, Z starts: ', np.linalg.norm(Y0), np.linalg.norm(R0), np.linalg.norm(Z0))
    Y = Y0
    R = R0
    Z = Z0

    for i in tqdm(range(maxit)):
        R_pre_proj = Vhat.T @ (Y + Z / beta) @ Vhat
        R_pre_proj = (R_pre_proj + R_pre_proj.T) / 2.

        S, U = np.linalg.eig(R_pre_proj)
        if not args.low_rank:
            pos_idx = S > 0
            if pos_idx.sum() > 0:
                vhat_u = Vhat @ U[:, pos_idx] # tempid
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
        Y = VRV - (L + Z) / beta
        Y = (Y + Y.T) / 2.
        Y[J] = 0
        Y[0, 0] = 1
        Y = np.minimum(1, np.maximum(0, Y))
        Y[0, 0] = 1
        Y[J] = 0
        Y[np.abs(Y) < tol] = 0

        pR = Y - VRV

        # update Z
        Z = Z + gamma * beta * (Y - VRV)
        Z = (Z + Z.T) / 2.
        Z[np.abs(Z) < tol] = 0
        # the computed lower bound is ...

        if i % 100 == 0:
            lbd = lower_bound(L, J, Vhat, Z, n, scale=normL)
            npr = np.linalg.norm(pR, 'fro')
            print(f'Epoch: {i:d} | Lower bound: {lbd:.4f} | norm pR: {npr:.4f}')

    tt = time.time() - st
    print('Done! | Elapsed: {:.2f}min'.format(tt / 60))

def lower_bound(L, J, Vhat, Z, n, scale=1):
    That = make_that(n)
    # print('That shape', That.shape)

    Q, _ = np.linalg.qr(That.T)
    Q = Q[:, :-1]
    # print('Q shape', Q.shape)

    Uloc = np.concatenate([Vhat, Q], 1)
    # print('Uloc shape', Uloc.shape)

    Zloc = Uloc.T @ Z @ Uloc
    # print('Zloc shape', Zloc.shape)

    # W12 = Zloc(1:(n-1)^2+1,(n-1)^2+2:end);
    # W22 = Zloc((n-1)^2+2:end,(n-1)^2+2:end);
    # W11 = Zloc(1:(n-1)^2+1,1:(n-1)^2+1);
    # W11 = (W11+W11')/2;
    W11 = Zloc[:(n-1)*(n-1)+1, :(n-1)*(n-1)+1]
    W11 = (W11 + W11.T) / 2.
    W12 = Zloc[:(n-1)*(n-1)+1, (n-1)*(n-1)+1:]
    W22 = Zloc[(n-1)*(n-1)+1:, (n-1)*(n-1)+1:]

    # Project W11 onto negative definite matrices for Zp
    Dw, Uw = np.linalg.eig(W11)
    neg_idx = Dw < 0
    W11 = (Uw[:, neg_idx] * Dw[neg_idx]) @ Uw[:, neg_idx].T

    Zp = Uloc @ np.block([[W11, W12], [W12.T, W22]]) @ Uloc.T
    Zp = (Zp + Zp.T) / 2

    # dont know why we dont just use Y_out
    # print('L shape: {} | n**2 + 1 = {}'.format(L.shape, n*n + 1))
    Yp = np.zeros(L.shape)
    Yp[L + Zp < 0] = 1
    Yp[J] = 0
    Yp[0, 0] = 1
    print(f'Z  norm: {np.linalg.norm(Z):.4f}')
    print(f'Zp norm: {np.linalg.norm(Zp):.4f}')
    print(f'Yp norm: {np.linalg.norm(Yp):.4f}')
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
    admm_qap(L, Vhat, J, args)

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
