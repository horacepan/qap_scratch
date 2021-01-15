import pdb
import os
import time
import argparse
import numpy as np
import scipy.io
from scipy.optimize import linear_sum_assignment
from pyscipopt import Model, quicksum

def load_qap(n, dataset="nug"):
    prefix = '/home/hopan/github/qap_scratch/data/'
    fname = os.path.join(prefix, f"{dataset}{n}.mat")
    m = scipy.io.loadmat(fname)
    return m['A'], m['B']

def pp_vars(model, var_dict):
    tot = 0
    for k, v in var_dict.items():
        val = model.getVal(v)
        tot += val
        if val > 0:
            pass
            #print(k, val)
    print("Total:", tot)

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

def kaufman_broeckx(A, B):
    '''
    Note: q_{ijkl} = A_ik B_jl
    '''
    setup_start = time.time()
    n = A.shape[0]
    m = Model("kaufman-broeckx")
    x = {}
    z = {}
    ceq = {}
    cleq = {}

    rowsum_A = A.sum(axis=1)
    rowsum_B = B.sum(axis=1)
    a = rowsum_A.reshape(-1, 1) @ rowsum_B.reshape(1, -1)

    for i in range(n):
        for j in range(n):
            x[i,j] = m.addVar(f"x[{i},{j}]", "I", lb=0, ub=1)
            z[i,j] = m.addVar(f"z[{i},{j}]", "C", lb=0)

    for l in range(n):
        m.addCons(quicksum(x[k, l] for k in range(n)) == 1)
        m.addCons(quicksum(x[l, k] for k in range(n)) == 1)

    for i in range(n):
        for j in range(n):
            ceq[i,j] = m.addCons(z[i,j] == x[i,j] * quicksum(A[i,k]*B[j,l]*x[k,l] for k in range(n) for l in range(n)))
            cleq[i,j] = m.addCons(z[i,j] >= quicksum(A[i,k]*B[j,l]*x[k,l] - a[i,j]*(1-x[i,j]) for k in range(n) for l in range(n)))

    m.setObjective(quicksum(z[i,j] for i in range(n) for j in range(n)))
    setup_end = time.time()
    opt_start = time.time()
    m.optimize()
    opt_end = time.time()

    m.printSol()
    print("Objective val: {:.2f}".format(m.getObjVal()))
    print(f"Setup time | {setup_end - setup_start:.2f}s")
    print(f"Opt time   | {opt_end - opt_start:.2f}s")

def kaufman_broeckx2(A, B):
    setup_start = time.time()
    n = A.shape[0]
    m = Model("kaufman-broeckx")
    x = {}
    z = {}
    ceq = {}
    cleq = {}

    rowsum_A = A.sum(axis=1)
    rowsum_B = B.sum(axis=1)
    a = rowsum_A.reshape(-1, 1) @ rowsum_B.reshape(1, -1)

    for i in range(n):
        for j in range(n):
            x[i,j] = m.addVar(f"x[{i},{j}]", "I", lb=0, ub=1)
            z[i,j] = m.addVar(f"z[{i},{j}]", "C", lb=0)
            m.addCons(x[i,j]*x[i,j] - x[i,j] == 0)
    #for j in range(n):
        #    m.addCons(quicksum(x[i,j] * x[i,j] for i in range(n)) == 1, f"col[{j}]")
        #for i in range(n):
        #    m.addCons(quicksum(x[i,j] * x[i,j] for j in range(n)) == 1, f"row[{i}]")

    for l in range(n):
        m.addCons(quicksum(x[k, l] for k in range(n)) == 1)
        m.addCons(quicksum(x[l, k] for k in range(n)) == 1)

    for i in range(n):
        for j in range(n):
            ceq[i,j] = m.addCons(z[i,j] == x[i,j] * quicksum(A[i,k]*B[j,l]*x[k,l] for k in range(n) for l in range(n)))
            cleq[i,j] = m.addCons(z[i,j] >= quicksum(A[i,k]*B[j,l]*x[k,l] - a[i,j]*(1-x[i,j]) for k in range(n) for l in range(n)))

    m.setObjective(quicksum(z[i,j] for i in range(n) for j in range(n)))
    setup_end = time.time()
    opt_start = time.time()
    m.optimize()
    opt_end = time.time()

    m.printSol()
    print("Objective val: {:.2f}".format(m.getObjVal()))
    print(f"Setup time | {setup_end - setup_start:.2f}s")
    print(f"Opt time   | {opt_end - opt_start:.2f}s")


def lawlers_linearization(A, B):
    setup_start = time.time()
    n = A.shape[0]
    m = Model("Lawlers Linearization")
    x = {}
    y = {}

    rowsum_A = A.sum(axis=1)
    rowsum_B = B.sum(axis=1)
    a = rowsum_A.reshape(-1, 1) @ rowsum_B.reshape(1, -1)

    for i in range(n):
        for j in range(n):
            x[i,j] = m.addVar(f"x[{i},{j}]", "I", lb=0, ub=1)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    y[i,j,k,l] = m.addVar(f"y[{i}{j}{k}{l}]", "I", 0, 1)
                    m.addCons(x[i,j] + x[k,l] >= 2*y[i,j,k,l])
    m.addCons(quicksum(y[i,j,k,l] for i in range(n) for j in range(n) for k in range(n) for l in range(n)) == (n*n))

    for l in range(n):
        m.addCons(quicksum(x[k, l] for k in range(n)) == 1)
        m.addCons(quicksum(x[l, k] for k in range(n)) == 1)

    m.setObjective(quicksum(A[i,k]*B[j,l]*y[i,j,k,l] for i in range(n) for j in range(n) for k in range(n) for l in range(n)))

    setup_end = time.time()
    opt_start = time.time()
    m.optimize()
    opt_end = time.time()

    m.printSol()
    print("Objective val: {:.2f}".format(m.getObjVal()))
    print(f"Setup time | {setup_end - setup_start:.2f}s")
    print(f"Opt time   | {opt_end - opt_start:.2f}s")
    pp_vars(m, y)
    if m.getStatus() == "optimal":
        print("Solved! Optimal value:", m.getObjVal())


def xin_yuan_linearization(A, B):
    setup_start = time.time()
    n = A.shape[0]
    L = _gen_glb_lmatrix(A, B)
    m = Model("Xin-Yuan Linearization")
    x = {}
    z = {}
    ceq = {}
    cleq = {}

    rowsum_A = A.sum(axis=1)
    rowsum_B = B.sum(axis=1)
    a = rowsum_A.reshape(-1, 1) @ rowsum_B.reshape(1, -1)

    for i in range(n):
        for j in range(n):
            x[i,j] = m.addVar(f"x[{i},{j}]", "I", lb=0, ub=1)
            z[i,j] = m.addVar(f"z[{i},{j}]", "C", lb=0)
            m.addCons(x[i,j]*x[i,j] - x[i,j] == 0)
    for l in range(n):
        m.addCons(quicksum(x[k, l] for k in range(n)) == 1)
        m.addCons(quicksum(x[l, k] for k in range(n)) == 1)

    for i in range(n):
        for j in range(n):
            ceq[i,j] = m.addCons(z[i,j] == x[i,j] * quicksum(A[i,k]*B[j,l]*x[k,l] for k in range(n) for l in range(n)))
            cleq[i,j] = m.addCons(z[i,j] >= quicksum(A[i,k]*B[j,l]*x[k,l] - a[i,j]*(1-x[i,j]) for k in range(n) for l in range(n)))

    m.setObjective(quicksum(z[i,j] for i in range(n) for j in range(n)))
    setup_end = time.time()
    opt_start = time.time()
    m.optimize()
    opt_end = time.time()

    m.printSol()
    print("Objective val: {:.2f}".format(m.getObjVal()))
    print(f"Setup time | {setup_end - setup_start:.2f}s")
    print(f"Opt time   | {opt_end - opt_start:.2f}s")


def main(args):
    print(args)
    A, B = load_qap(args.n, args.dataset)
    #qap_model(A, B)
    if args.version == 'kb2':
        print("Running Kaufman Broeckx 2")
        kaufman_broeckx2(A, B)
    elif args.version == 'lawler':
        print("Running Lawlers Linearization")
        lawlers_linearization(A, B)
    else:
        print("Running Kaufman Broeckx")
        kaufman_broeckx(A, B)
    #A = np.random.random((args.n, args.n))
    #linear_assignment(A)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--dataset", type=str, default="nug")
    parser.add_argument("--version", type=str, default="kb")
    args = parser.parse_args()

    main(args)
