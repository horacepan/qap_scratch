import pdb
import time
import random
import numpy as np
from snpy.perm import Perm

def vec(p):
    return np.reshape(p, (-1, 1), order='F')

def outer(p):
    vp = vec(p)
    return vp @ vp.T

def kron(p):
    return np.kron(p, p)

def random_perm(n, iters=100):
    gens = [Perm.trans(1, 2, n), Perm.cycle(1, n, n)]
    p = Perm.eye(n)

    for _ in range(iters):
        p = p * random.choice(gens)

    return p

def k2v(kp, vp):
    '''
    Convert kron -> vec vec.T
    '''
    output = np.zeros(kp.shape)
    n = int(kp.shape[0] ** 0.5)
    for i in range(n):
        for j in range(n):
            row_idx = i + (j * n)
            block = kp[i*n: (i+1)*n, j*n: (j+1)*n]
            output[row_idx] = block.reshape(1, -1, order='F')
    return output

def v2k(kp, vp):
    '''
    Convert vec vec.T -> kron
    '''
    output = np.zeros(kp.shape)
    n = int(kp.shape[0] ** 0.5)
    for i in range(n):
        for j in range(n):
            row_idx = i + (j * n)
            row = vp[row_idx]
            output[i*n: (i+1)*n, j*n: (j+1)*n] = row.reshape(n, n, order='F')
    return output

def main():
    n = 30
    p = random_perm(n)
    pm = p.mat()

    kp = kron(pm)
    vp = outer(pm)
    st = time.time()
    kp2v = k2v(kp, vp)
    end = time.time()

    vst = time.time()
    vp2k = v2k(kp, vp)
    vend = time.time()
    print('Convert kron to vec : {} | Elapsed: {:.3f}s'.format(np.allclose(kp2v, vp), 2 * (end - st)))
    print('Convert vec  to kron: {} | Elapsed: {:.3f}s'.format(np.allclose(vp2k, kp), 2 * (vend - vst)))
    pdb.set_trace()

if __name__ == '__main__':
    main()