import pdb
import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import scipy.io
import scipy.optimize
import matplotlib.pyplot as plt

from mat_utils import v2k, k2v, random_perm
from snpy.utils import hook_length
# from fourier_admm import make_bdiag_mask

PREFIX = '../data/datasets1/'

def make_bdiag_mask(n):
    irreps = [
        (n - 2, 1, 1),
        (n - 2, 2),
        (n - 1, 1),
        (n - 1, 1),
        (n - 1, 1),
        (n,),
        (n,),
    ]
    hls = [int(hook_length(p)) for p in irreps]
    size = sum(hls)
    mask = np.zeros((size, size))

    idx = 0
    for l in hls:
        mask[idx: idx+l, idx:idx+l] = 1
        idx += l

    return mask

def fnorm(m):
    return np.linalg.norm(m, 'fro')

def pnorm(m, msg):
    print(f'{msg} | Norm: {fnorm(m):.2f} | ASum: {np.abs(m).sum():.2f}')

def psd_project(mat):
    S, U = np.linalg.eigh(mat)
    pos_idx = S > 0
    return (U[:, pos_idx] * S[pos_idx]) @ U[:, pos_idx].T

def make_L(A, B, C=None):
    if C is None:
        cvec = np.zeros(A.shape[0]*A.shape[1])
    else:
        cvec = np.reshape(C, -1, 'F')

    Ldim = 1 + A.shape[0] * A.shape[0]
    L = np.zeros((Ldim, Ldim))
    L[0, 1:] = cvec
    L[1:, 0] = cvec
    L[1:, 1:] = np.kron(B, A)
    return L

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

def make_vhat(n):
    V = np.concatenate([np.eye(n - 1), -np.ones((1, n - 1))])
    V, _ = np.linalg.qr(V, 'reduced')

    n, n1 = V.shape
    vxv = np.kron(V, V)
    r1 = np.zeros((1, n1 * n1 + 1))
    r1[0, 0] = np.sqrt(0.5)

    r2 = np.concatenate([np.ones((n * n, 1)) * (np.sqrt(0.5) / n), vxv], axis=1)
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

def admm_qap(A, B, C, args):
    '''
    A: Flow matrix
    B: Distance matrix
    C: linear cost matrix
    args: maxit, tol, gamma, low_rank
    '''
    n = A.shape[0]
    L = make_L(A, B)
    Vhat = make_vhat(n)
    J = make_gangster(n)

    st = time.time()
    maxit = args.maxit
    tol = args.tol
    gamma = args.gamma
    low_rank = args.low_rank

    normL = np.linalg.norm(L)
    Vhat_nrows = Vhat.shape[0]
    L = L * (n*n / normL)
    beta = n / 3.

    Y0 = make_y0(n)
    R0 = make_r0(n, Vhat, Y0)
    Z0 = Y0 - (Vhat @ R0 @ Vhat.T)

    Y = Y0
    R = R0
    Z = Z0
    lbd = -float('inf')
    ubd = float('inf')

    for i in range(maxit):
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
        Y = VRV - ((L + Z) / beta)
        Y = (Y + Y.T) / 2.
        Y[J] = 0; Y[0,0] = 1
        Y = np.minimum(1, np.maximum(0, Y))
        Y[J] = 0; Y[0,0] = 1
        Y[np.abs(Y) < tol] = 0
        pR = Y - VRV

        # update Z
        Z = Z + gamma * beta * (Y - VRV)
        Z = (Z + Z.T) / 2.
        Z[np.abs(Z) < tol] = 0

        if i % 100 == 0:
            scale = normL / (n * n)
            lbd = max(lbd, lower_bound(L, J, Vhat, Z, n, scale=scale))
            ubd = min(ubd, upper_bound(Y, A, B))
            npr = get_ubd(random_perm(n).mat(), A, B)
            print(f'Iter {i} | Lower bound: {lbd:.2f} | Upper bound: {ubd:.2f} | Rand: {npr:.2f}')

    tt = time.time() - st
    print('Done! | Lbd: {:.2f} | Elapsed: {:.2f}min'.format(lbd, tt / 60))
    return

def lower_bound(L, J, Vhat, Z, n, scale=1):
    That = make_that(n)
    Q, _ = np.linalg.qr(That.T)
    Q = Q[:, :-1]

    Uloc = np.concatenate([Vhat, Q], 1)
    Zloc = Uloc.T @ Z @ Uloc

    W11 = Zloc[:(n-1)*(n-1)+1, :(n-1)*(n-1)+1]
    W11 = (W11 + W11.T) / 2.
    W12 = Zloc[:(n-1)*(n-1)+1, (n-1)*(n-1)+1:]
    W22 = Zloc[(n-1)*(n-1)+1:, (n-1)*(n-1)+1:]

    # Project W11 onto negative definite matrices for Zp
    Dw, Uw = np.linalg.eigh(W11)
    neg_idx = Dw < 0
    W11 = (Uw[:, neg_idx] * Dw[neg_idx]) @ Uw[:, neg_idx].T

    Zp = Uloc @ np.block([[W11, W12], [W12.T, W22]]) @ Uloc.T
    Zp = (Zp + Zp.T) / 2

    Yp = np.zeros(L.shape)
    Yp[L + Zp < 0] = 1
    Yp[J] = 0; Yp[0, 0] = 1
    lbd = ((L + Zp) * Yp).sum() * scale
    return lbd

def upper_bound(Y, A, B):
    n = int((Y.shape[0] - 1) ** 0.5)
    Yloc = (Y + Y.T) / 2.
    D, U = np.linalg.eigh(Y)
    d = D[-1]
    u = np.reshape(U[:, -1], (-1, 1), 'F')
    Yloc_hat = (d * u @ u.T)
    Xloc_hat = np.reshape(Yloc_hat[1:, 0], (n, n), 'F')

    en = np.ones((n, 1))
    In = np.eye(n)
    ein = np.kron(en.T, In)
    ien = np.kron(In, en.T)
    A_eq = np.concatenate([ein, ien])[:-1, :]
    b_eq = np.ones((2 * n, 1))[:-1, :]
    bounds = [0, 1]
    c = np.reshape(Xloc_hat, -1, 'F')
    eig_res = scipy.optimize.linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
        method='simplex'
    )

    eig_X = np.reshape(eig_res.x, (n, n), 'F')
    eig_ubd = get_ubd(eig_X, A, B)

    Yloc_hat = Yloc
    Xloc_hat = np.reshape(Yloc_hat[1:, 0], (n, n), 'F')
    col_res = scipy.optimize.linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
        method='simplex'
    )
    col_X = np.reshape(col_res.x, (n, n), 'F')
    col_ubd = get_ubd(col_X, A, B)

    return min(eig_ubd, col_ubd)

def get_ubd(simplex_X, A, B):
    return np.trace(A @ simplex_X @ B @ simplex_X.T)

def main(args):
    np.random.seed(args.seed)
    fname = os.path.join(PREFIX, args.ex + '.mat')
    mats = scipy.io.loadmat(fname)

    A = mats['A']
    B = mats['B']
    admm_qap(A, B, C=None, args=args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ex', type=str, default='nug12')
    parser.add_argument('--maxit', type=int, default=1000)
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--low_rank', action='store_true', default=False)
    parser.add_argument('--gamma', type=float, default=1.618)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
