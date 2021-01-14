import pdb
import os
import argparse
import numpy as np
import scipy.io
from scipy.optimize import linear_sum_assignment
from pyscipopt import Model, quicksum

def make_perm(rows, cols):
    n = len(rows)
    mat = np.zeros((n, n))
    for r, c in zip(rows, cols):
        mat[r, c] = 1
    return mat

def load_nug(n):
    prefix = '/home/hopan/github/qap_scratch/data/'
    fname = os.path.join(prefix, f"nug{n}.mat")
    m = scipy.io.loadmat(fname)
    return m['A'], m['B']

def qap_model(A, B):
    n, _ = A.shape
    model = Model("qap")
    x = {}

    # create the variables
    for i in range(n):
        for j in range(n):
            x[i, j] = model.addVar(name=f"x[{i},{j}]", vtype="I", lb=0, ub=1)

    # create constraint: pairs of elements in a row or col mult == 0
    for j in range(n):
        for k in range(j):
            model.addCons(x[i,j] * x[i,k] == 0, f"x[{i},:]")
            model.addCons(x[j,i] * x[k,i] == 0, f"x[:,{i}]")

    # x_ij^2 - x_ij == 0
    for i in range(n):
        for j in range(n):
            model.addCons(x[i,j] * x[i,j] - x[i,j] == 0, f"x[{i},{j}]^2")

    # \sum_i x_ij^2 - 1 == 0
    for j in range(n):
        model.addCons(quicksum(x[i,j] * x[i,j] for i in range(n)) - 1 == 0, f"col[{j}]")

    for i in range(n):
        model.addCons(quicksum(x[i,j] * x[i,j] for j in range(n)) - 1 == 0, f"row[{i}]")


    model.setObjective(quicksum(x[i, j] * A[i, j] * B[k, l] * x[k, l] for i in range(n) for j in range(n) for k in range(n) for l in range(n)))
    model.optimize()
    model.printSol()
    pdb.set_trace()
    print(model.ObjVal)

def linear_assignment(A):
    n, _ = A.shape
    x = {}
    m = Model("lap")
    model = m

    # create the variables
    for i in range(n):
        for j in range(n):
            x[i, j] = m.addVar(f"x[{i},{j}]", vtype="I", lb=0, ub=1)

    # create constraint: pairs of elements in a row or col mult == 0
    for j in range(n):
        for k in range(j):
            model.addCons(x[i,j] * x[i,k] == 0, f"x[{i},:]")
            model.addCons(x[j,i] * x[k,i] == 0, f"x[:,{i}]")

    # x_ij^2 - x_ij == 0
    for i in range(n):
        for j in range(n):
            model.addCons(x[i,j] * x[i,j] - x[i,j] == 0, f"x[{i},{j}]^2")

    # \sum_i x_ij^2 - 1 == 0
    for j in range(n):
        model.addCons(quicksum(x[i,j] * x[i,j] for i in range(n)) - 1 == 0, f"col[{j}]")

    for i in range(n):
        model.addCons(quicksum(x[i,j] * x[i,j] for j in range(n)) - 1 == 0, f"row[{i}]")


    m.setObjective(quicksum(x[i, j] * A[i, j] for i in range(n) for j in range(n)))
    m.optimize()
    m.printSol()
    print("===========================")

    rows, cols = linear_sum_assignment(A)
    pmat = make_perm(rows, cols)
    sol = (A * pmat).sum()
    print(f"Linear sum assignment sol: {sol:.2f}")

def main(args):
    print(args)
    A, B = load_nug(args.n)
    qap_model(A, B)
    #A = np.random.random((args.n, args.n))
    #linear_assignment(A)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    args = parser.parse_args()

    main(args)
