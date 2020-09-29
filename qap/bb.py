import pdb
import time
import os
import random
import argparse
from collections import deque
import numpy as np
import scipy.io
from admm import admm_qap # A, B, args
from bb_admm import bb_admm
from mat_utils import brute_solve
PREFIX = '../data/datasets1'

def sample_diff(base, exl):
    extras = [i for i in base if i not in exl]
    return random.choice(extras)

def bb(A, B, C, args):
    n = A.shape[0]
    opt_lb, opt_ub = admm_qap(A, B, C, args=args)
    print(f'Init global opts: ({opt_lb}, {opt_ub})')
    sols = []

    if np.allclose(opt_lb, opt_ub):
        return [(opt_lb, 'top')]

    facs = [i for i in range(n)]
    locs = [i for i in range(n)]
    X = np.zeros(A.shape)
    node = {
        'h': 0,
        'lb': opt_lb,
        'ub': opt_ub,
        'X': X,
    }
    q = deque([], maxlen=args.max_qsize)
    q.append(node)

    while len(q) > 0:
        if args.bfs:
            node = q.popleft() #
        elif args.dfs: # dfs
            node = q.pop() #
        else:
            print('Must be dfs or bfs')
            exit()

        # lb <= opt.ub. Find the assignments
        rows, cols = np.where(node['X'] == 1)
        urows = [i for i in facs if i not in rows]
        avail_locs = [i for i in locs if i not in cols]
        i = random.choice(urows)
        i = 9

        # randomly sample an available location
        ubs = np.zeros(len(avail_locs))
        lbs = np.zeros(len(avail_locs))
        assignments = []

        for idx, j in enumerate(avail_locs):
            _lb, _ub, Xa = bb_admm(A, B, C, opt_ub, node['X'], i, j, args)
            lbs[idx] = _lb
            ubs[idx] = _ub
            assignments.append(Xa)

        opt_ub = min(opt_ub, np.min(ubs[ubs < float('inf')]))
        opt_lb = max(opt_lb, np.max(lbs[lbs > -float('inf')]))
        idb = np.where(lbs <= opt_ub)[0]
        #print('Mapped to: {}'.format(idb))
        #print('curr node h: {} | lbs: {}'.format(node['h'], lbs))
        #print('curr node h: {} | ubs: {}'.format(node['h'], ubs))
        print('==============')

        for _id in idb:
            node = {
                'lb': lbs[_id],
                'ub': ubs[_id],
                'X': assignments[_id],
                'h': node['h'] + 1
            }

            if node['lb'] == node['ub']:
                sols.append((node['lb'], node['X']))
            else:
                q.append(node)
            print(node['lb'], node['ub'])

    return sols

def main(args):
    st = time.time()
    np.random.seed(args.seed)
    mats = scipy.io.loadmat(os.path.join(PREFIX, args.ex + '.mat'))
    A = mats['A']
    B = mats['B']
    C = np.zeros(A.shape)

    # A = np.random.random((3,3))
    # A = np.round(A + A.T, 3)
    # B = np.array([[2, 1, 1],
    #               [1, 2, 2],
    #               [1, 2, 3]])
    # C = np.zeros(A.shape)
    # opt_sol = brute_solve(A, B, C)
    # print(opt_sol)
    sols = bb(A, B, C, args)
    print('Solutions:')
    for s in sols:
        print(s[0])
    print('Elapsed: {:.2f}s'.format(time.time() -st))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ex', type=str, default='nug12')
    parser.add_argument('--maxit', type=int, default=1000)
    parser.add_argument('--max_qsize', type=int, default=1000)
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--low_rank', action='store_true', default=False)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=1.618)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bfs', action='store_true', default=False)
    parser.add_argument('--dfs', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    main(args)

