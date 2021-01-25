import sys
import pdb
import numpy as np

class Simplex:
    def __init__(self, A, b, c, const=0, mode="maximize"):
        A_s, c_s = self._add_slack_vars(A, c)
        self.A = A
        self.b = b
        self.c = c
        self.const = const
        self.mode = mode

        self.A_tableau = A_s
        self.c_tableau = c_s
        self.b_tableau = b
        self.tableau_const = const
        self.basic_vars = list(range(A.shape[1], self.A_tableau.shape[1]))
        self._sol = np.zeros(len(c_s))
        self._sol[A.shape[1]:] = self.b_tableau

    def _add_slack_vars(self, A, c):
        nslack = A.shape[0]
        c_new = np.append(c, np.zeros(nslack))
        A_new = np.concatenate([A, np.eye(nslack)], axis=1)
        return A_new, c_new

    def pivot(self):
        if self.mode == "maximize":
            i, j = self.find_pivot_max()
        elif self.mode == "minimize":
            i, j = self.find_pivot_min()
        else:
            raise Exception(f"{self.mode} is not a valid mode for simplex")

        #print(f"{self.basic_vars[i]+1} is leaving, {j+1} is entering")

        constraint = i
        try:
            leaving_var = self.basic_vars[i]
        except:
            pdb.set_trace()
        entering_var = j

        self.basic_vars[i] = j
        a_ij = self.A_tableau[i, j]
        row_i = self.A_tableau[i, :]
        row_i_normed = row_i / a_ij
        b_i_normed = self.b_tableau[i] / a_ij

        for k in range(self.A_tableau.shape[0]):
            if k == i:
                self.A_tableau[i] = row_i_normed
                self.b_tableau[i] = b_i_normed
            else:
                row = self.A_tableau[k, :]
                a_kj = self.A_tableau[k, j]
                self.A_tableau[k] -= a_kj*row_i_normed
                self.b_tableau[k] -= a_kj*b_i_normed

        # do the same for the cost vector
        c_j = self.c_tableau[j]
        self.c_tableau -= c_j * row_i_normed
        self.tableau_const -= c_j*b_i_normed

        self._sol[:] = 0
        for idx, v in enumerate(self.basic_vars):
            self._sol[v] = self.b_tableau[idx]

    def find_pivot_max(self, rule='lex'):
        idx = None
        for i, _c in enumerate(self.c_tableau):
            if _c > 0:
                idx = i
                break

        # find the row i such that b_i / a_ij is minimal
        eps = float("inf")
        min_row = None
        for row_idx in range(self.A_tableau.shape[0]):
            if self.A_tableau[row_idx, idx] > 0:
                val = self.b_tableau[row_idx] / self.A_tableau[row_idx, idx]
                if val < eps:
                    min_row = row_idx
                    eps = val

        return min_row, idx

    def find_pivot_min(self):
        idx = None
        for i, _c in enumerate(self.c_tableau):
            if _c < 0:
                idx = i
                break

        # find the row i such that b_i / a_ij is minimal
        eps = float("inf")
        min_row = None
        for row_idx in range(self.A_tableau.shape[0]):
            if self.A_tableau[row_idx, idx] > 0:
                val = self.b_tableau[row_idx] / self.A_tableau[row_idx, idx]
                if val < eps:
                    min_row = row_idx
                    eps = val

        return min_row, idx

    def obj_val(self):
        return (self._sol[:len(self.c)] * self.c).sum() + self.const

    def sol(self):
        return self._sol[:len(self.c)]

    def sol2(self):
        sol = np.zeros(len(self.c))

        for idx, i in enumerate(self.basic_vars):
            if i < len(sol):
                sol[i] = self.b_tableau[idx]
        return sol

    def add_cut(self, cut):
        # this may expand the tableau
        pass

    def current_tableau(self):
        return self._tableau

    def constraints(self):
        return self.A, self.b

    def objective(self):
        return self.c

    def solve(self, verbose=False):
        sol = 0
        iters = 0
        if verbose:
            self.ppt()

        if self.mode == "maximize":
            while not np.all(self.c_tableau <= 0):
                self.pivot()
                if verbose:
                    self.ppt()
                iters += 1
        elif self.mode == "minimize":
            while not np.all(self.c_tableau >= 0):
                self.pivot()
                if verbose:
                    self.ppt()
                iters += 1
        return self.tableau_const

    def add_constraint(self, ai, bi):
        '''
        Add new constraint to the problem
        '''
        A_new = np.append(self.A, a_i.reshape(1, -1))
        b_new = np.append(self.b, bi)

    def ppt(self):
        print('const:', self.tableau_const)
        print('c:')
        print(self.c_tableau)
        print('A:')
        for row in self.A_tableau:
            print(row)
        print('b:')
        print(self.b_tableau)

        sol_vec = np.zeros(self.A_tableau.shape[1])
        for idx, i in enumerate(self.basic_vars):
            try:
                sol_vec[i] = self.b_tableau[idx]
            except:
                pdb.set_trace()
        print("Solution vec:", self._sol)
        print("Objective: ", self.obj_val())
        print("==============================")
