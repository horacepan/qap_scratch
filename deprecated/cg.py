import pdb
import numpy as np
from scipy.linalg import null_space
from snpy.perm import Perm
from snpy.sn_irrep import SnIrrep

def in_null(mat, vec):
    return np.allclose(0, np.matmul(mat, vec.reshape((-1, 1))))

def kron_irreps(p1, p2, n):
    '''
    p1: a partition n
    p2: a partition n
    Return list of tuples (mu, m_mu)
        where m_mu is the multiplicity of mu in p1 \otimes p2

    (n-1, 1) \otimes (n-2, 1, 1) = (n-1, 1), (n-2, 2), (n-2, 1, 1), (n-3, 2, 1), (n-3, 1, 1, 1)
    '''
    if p1 == (n-1, 1) and p2 == (n-1, 1):
        res = [(n,), (n-1, 1), (n-2, 2), (n-2, 1, 1)]
    elif p1 == (n-1, 1) and p2 == (n-2, 2):
        res = [(n-1, 1), (n-2, 2), (n-2, 1, 1), (n-3, 3), (n-3, 2, 1)]
    elif  p1 == (n-1, 1) and p2 == (n-2, 1, 1):
        res = [(n-1, 1), (n-2, 2), (n-2, 1, 1), (n-3, 2, 1), (n-3, 1, 1, 1)]
    else:
        raise Exception(f"Multiplicity not implemented for {p1}, {p2}")

    return res

def block_irrep(g, irreps):
    mats = []
    sz = 0
    for p in irreps:
        rho = SnIrrep(p)
        mats.append(rho(g))
        sz += mat.shape[0]

    # now contruct
    blocked = np.zeros((sz, sz))
    idx = 0
    for mat in mats:
        d = mat.shape[0]
        blocked[idx:idx+d, idx:idx+d]
        idx += d

    return blocked

def random_null_vec(mat, dim):
    '''
    mat: n x k matrix
    Return: random vector in the nullspace of mat
    '''
    nvecs = null_space(mat)
    nbasis = nvecs[:, -dim:]
    weights = (np.random.random(dim) * 2) - 1 # [-1, 1]
    vec = (nbasis * weights).sum(axis=1)
    return vec

def kron_factor_id(mat, dim):
    size = mat.shape[0] // dim
    m = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            _i = i * dim
            _j = j * dim
            m[i, j] = mat[_i, _j]
    return m

def normalize_rows(mat):
    row_norms = np.sqrt(np.square(mat).sum(axis=1))
    return mat / row_norms[:, np.newaxis]

def intertwine(p1, p2, n, verbose=True):
    irreps = kron_irreps(p1, p2, n)

    g1 = Perm.trans(1, 2, n)
    g2 = Perm.cycle(1, n, n)

    rho1 = SnIrrep(p1)
    rho2 = SnIrrep(p2)

    g1_p1xp2 = np.kron(rho1(g1), rho2(g1))
    g2_p1xp2 = np.kron(rho1(g2), rho2(g2))
    g1_blocked = rho1(g1) #block_irrep(g1, irreps)
    g2_blocked = rho2(g2) #block_irrep(g2, irreps)

    mult = 1
    d = g1_p1xp2.shape[0]
    dnu = g1_blocked.shape[0] // mult
    eye = np.eye(g1_p1xp2.shape[0])
    eye_zd = np.eye(g1_blocked.shape[0] * mult)
    k1 = np.kron(eye, g1_blocked) - np.kron(g1_p1xp2, eye_zd)
    k2 = np.kron(eye, g2_blocked) - np.kron(g2_p1xp2, eye_zd)
    k = np.concatenate([k1, k2], axis=0)


    rand_null = random_null_vec(k, mult * mult)
    print('random vec shape', rand_null.shape)
    R = np.reshape(rand_null, (mult * dnu, d), 'F') # this reshapes columnwise
    RRT = np.matmul(R, R.T)

    M = kron_factor_id(RRT, dnu)
    _, evecs = np.linalg.eig(M)
    S = np.kron(evecs, np.eye(dnu))
    output = normalize_rows(np.matmul(S.T, R))
    print('output shape', output.shape)

    if verbose:
        print('rand null in null', in_null(k, rand_null))
        print('R intertwines g1?', np.allclose(np.matmul(R, g1_p1xp2), np.matmul(g1_blocked, R)))
        print('R intertwines g2?', np.allclose(np.matmul(R, g2_p1xp2), np.matmul(g2_blocked, R)))
        print('rrt is commutant g1 block?', np.allclose(np.matmul(RRT, g1_blocked), np.matmul(g1_blocked, RRT)))
        print('rrt is commutant g2 block?', np.allclose(np.matmul(RRT, g2_blocked), np.matmul(g2_blocked, RRT)))

    return output

if __name__ == '__main__':
    n = 8
    p1 = (n-1, 1)
    p2 = (n-1, 1)

    intertwine(p1, p2, n)
