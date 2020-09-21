import pdb
import time
import random
from tqdm import tqdm

import numpy as np

from snpy.perm import Perm, sn
from snpy.sn_irrep import SnIrrep

from sylvester import rand_perm
from cg_utils import intertwine

def main():
    n = 12
    g1 = Perm.trans(1, 2, n)
    g2 = Perm.cycle(1, n, n)
    k1 = np.kron(g1.mat(), g1.mat())
    k2 = np.kron(g2.mat(), g2.mat())

    irrep_mults = {
        (n - 2, 1, 1): 1,
        (n - 2, 2)   : 1,
        (n - 1, 1)   : 3,
        (n,)         : 2,
    }
    irreps = {i: SnIrrep(i) for i in irrep_mults.keys()}
    intws = {}

    _st = time.time()
    for part, mult in irrep_mults.items():
        st = time.time()
        rho = irreps[part]
        mult = irrep_mults[part]
        c = intertwine(k1, k2, rho(g1), rho(g2), mult)
        intws[part] = c
        end = time.time()
        print('Done with {} | Part elapsed: {:.2f} | Total elapsed: {:.2f}s'.format(
            part, end -st, time.time() - _st))

    pdb.set_trace()
if __name__ == '__main__':
    main()
