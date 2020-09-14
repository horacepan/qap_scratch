import pdb
import numpy as np
import cvxpy as cp
from snpy.perm import Perm, sn
from snpy.sn_irrep import SnIrrep

from marginals import *
from test import *
from ft import *

def gen_block_diag_vars(block_dict, irreps):
    d = int(sum([hook_length(i) for i in irreps]))
    big_mat = np.zeros((d,d))
    idx = 0
    blocks = []

    for irr in irreps:
        k = block_dict[irr].shape[0]

        mats = []
        if idx > 0:
            pre_zeros = np.zeros((k, idx))
            mats.append(pre_zeros)

        mats.append(block_dict[irr])

        if idx + k < d:
            post_zeros = np.zeros((k, d - idx - k))
            mats.append(post_zeros)

        blocks.append(mats)
        idx += k

    big_mat = cp.bmat(blocks)
    return big_mat

def make_variable(shape):
    return cp.Variable(shape)

def main(mat):
    amat = mat['A']
    bmat = mat['B']

    print('Input shape:', amat.shape)
    n = amat.shape[0]
    qap_func = gen_qap_func(amat, bmat)

    qap_fhats = {(n,): np.array([1])}
    variables = {(n,): cp.Variable((1, 1))}
    #qap_irreps = [(n - 1, 1), (n - 2, 2), (n - 2, 1, 1)]
    qap_irreps = [(n - 1, 1), (n - 2, 1, 1)]
    for irrep in [(n - 1, 1), (n - 2, 2), (n - 2, 1, 1)]:
        qhat = qap_fhat(amat, bmat, irrep)
        qap_fhats[irrep] = qhat
        variables[irrep] = make_variable(qhat.shape)

    c_lambdas = {}
    c_lambdas_inv = {}
    for irrep in [(n - 1, 1), (n - 2, 1, 1)]:
        c = c_lambda(irrep)
        c_lambdas_inv[irrep] = np.linalg.inv(c)
        c_lambdas[irrep] = c
        print(irrep, np.linalg.norm(c), np.allclose(c@c.T, np.eye(len(c))))
    pdb.set_trace()
    # These need to be functions of variables
    block_diags = {}
    #for irrep in qap_irreps:
    #for irrep in [(n - 1, 1), (n - 2, 2), (n - 2, 1, 1)]:
    for irrep in [(n - 1, 1), (n - 2, 1, 1)]:
        birreps, _ = qap_decompose(irrep)
        print('irrep: {} | decompose: {}'.format(irrep, birreps))
        block_diags[irrep] = gen_block_diag_vars(variables, birreps)
        print('done irrep: {} | decompose: {}'.format(irrep, birreps))

    # constraints are functions of the block_diags
    d_n     = hook_length((n,)) / math.factorial(n)
    d_n1    = hook_length((n - 1, 1)) / math.factorial(n)
    d_n22   = hook_length((n - 2, 2)) / math.factorial(n)
    d_n211  = hook_length((n - 2, 1, 1)) / math.factorial(n)
    obj = \
        d_n    * cp.sum(cp.multiply(variables[(n,)], qap_fhats[(n,)])) + \
        d_n1   * cp.sum(cp.multiply(variables[(n - 1, 1)], qap_fhats[(n - 1, 1)])) + \
        d_n22  * cp.sum(cp.multiply(variables[(n - 2, 2)], qap_fhats[(n - 2, 2)])) + \
        d_n211 * cp.sum(cp.multiply(variables[(n - 2, 1, 1)], qap_fhats[(n - 2, 1,  1)]))

    n1_one = np.ones((block_diags[(n-1, 1)].shape[0], 1))
    n2_one = np.ones((block_diags[(n-2, 1, 1)].shape[0], 1))
    constraints = [
        #c_lambdas_inv[(n - 1, 1)] @ block_diags[(n - 1, 1)] @ c_lambdas[(n - 1, 1)] >= 0,
        #(c_lambdas_inv[(n - 1, 1)] @ block_diags[(n - 1, 1)] @ c_lambdas[(n - 1, 1)]) @ n1_one  == 1,
        #(c_lambdas_inv[(n - 1, 1)] @ block_diags[(n - 1, 1)] @ c_lambdas[(n - 1, 1)]).T @ n1_one  == 1,
        # c_lambdas_inv[(n - 2, 2)] @ block_diags[(n - 2, 2)] @ c_lambdas[(n - 2, 2)] >= 0,
        c_lambdas_inv[(n - 2, 1, 1)] @ block_diags[(n - 2, 1, 1)] @ c_lambdas[(n - 2, 1, 1)] >= 0,
        #c_lambdas_inv[(n - 2, 1, 1)] @ block_diags[(n - 2, 1, 1)] @ c_lambdas[(n - 2, 1, 1)] @ n2_one == 1,
        #(c_lambdas_inv[(n - 2, 1, 1)] @ block_diags[(n - 2, 1, 1)] @ c_lambdas[(n - 2, 1, 1)]).T @ n2_one == 1,
        variables[(n,)] == 1
    ]
    print(len(constraints))
    problem = cp.Problem(cp.Maximize(obj), constraints)
    #problem.solve(solver=cp.OSQP)
    result =problem.solve(solver=cp.OSQP, verbose=True)
    #print(f'Obj: {problem.value:.2f}')
    perm = c_lambdas_inv[(n-1, 1)] @ block_diags[(n - 1, 1)].value @c_lambdas[(n-1, 1)]
    perm_tup = c_lambdas_inv[(n-2, 1, 1)] @ block_diags[(n - 2, 1, 1)].value @c_lambdas[(n-2, 1, 1)]
    res = ((perm @ amat @ perm.T)@ bmat).sum()
    print('Perm tup sums:', perm_tup.sum(axis=1), perm_tup.sum(axis=0), perm_tup.sum(), perm_tup.shape)
    print(f'Lower bound:  {res} | result: {result}')
    pdb.set_trace()

if __name__ == '__main__':
    #fname = '../data/rou12.mat'
    #mat = scipy.io.loadmat(fname)
    np.random.seed(0)
    n = 6
    mat = {}
    mat['A'] = np.random.randint(0, 10, (n,n))
    mat['B'] = np.random.randint(0, 100, (n,n))
    best = (mat['A'] * mat['B']).sum()
    best_p = Perm.eye(n)
    avg = 0

    for p in sn(mat['A'].shape[0]):
        obj = ((p.mat() @ mat['A'] @ p.mat().T) @ mat['B']).sum()
        avg += obj
        if obj > best:
            best = obj
            best_p = p
    avg = avg / math.factorial(n)
    print('Opt value: {} | Avg value: {:.2f} | Opt perm: {}'.format(best, avg, best_p))

    start = time.time()
    main(mat)
    print('Elapsed: {:.2f}s'.format(time.time() - start))
