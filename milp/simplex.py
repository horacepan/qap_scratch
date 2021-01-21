import pdb
import numpy as np

class Simplex:
    def __init__(self, A, b, c, const=0):
        A_s, c_s = self._add_slack_vars(A, c)
        self.A = A
        self.b = b
        self.c = c
        self.const = 0

        self.A_tableau = A_s
        self.c_tableau = c_s
        self.b_tableau = b
        self.tableau_const = const
        #self._nonbasic_vars = [i for i in range(A.shape[0])]
        #self._basic_vars = [i for i in range(A.shape[0], self.A_tableau[0])] # start off as the slack variables
        self.basic_vars = list(range(A.shape[0], self.A_tableau.shape[0]))
        self.nonbasic_vars = list(range(A.shape[0]))

    def _add_slack_vars(self, A, c):
        nslack = A.shape[0]
        c_new = np.append(c, np.zeros(nslack))
        A_new = np.concatenate([A, np.eye(nslack)], axis=1)
        return A_new, c_new

    def pivot(self):
        i, j = self.find_pivot_max()
        print("Pivoting on x[", j, "] x[j] =", self.c_tableau[j], "using row:", i, "aij:", self.A_tableau[i, j])
        a_ij = self.A_tableau[i, j]
        row_i = self.A_tableau[i, :]
        row_i_normed = row_i / a_ij
        b_i_normed = self.b_tableau[i] / a_ij
        #print("a_ij:", a_ij, "b normed:", b_i_normed)

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

        print("finding best pivot", self.A_tableau[:, idx])
        return min_row, idx

    def find_pivot_min(self):
        pass

    def _do_basis(self, basis, A, b, c):
        A_B = A[:, basis]
        A_B_T = A_B.T
        A_B_inv = np.linalg.inv(A_B)
        A_B_T_inv = np.linalg.inv(A_B_T)

        y = A_B_T_inv @ c[basis]
        b_bar = A_B_inv @ b
        A_bar = A_B_inv @ A

    def current_tableau(self):
        return self._tableau

    def constraints(self):
        return self.A, self.b

    def objective(self):
        return self.c

    def bfs(self):
        pass

    def solve(self):
        sol = 0
        iters = 0
        while not np.all(self.c_tableau <= 0):
            self.pivot()
            self.ppt()
            iters += 1
        print(f"Solved after {iters} iters")

    def pp(self):
        for ridx, row in enumerate(self._tableau):
            line = ""
            if ridx == 0:
                # first row of the tableau is the objective row
                continue

            for idx, val in enumerate(row):
                if idx > 0:
                    line += f"+ {val}x{idx+1}"
                elif idx < len(row) - 1:
                    line += f"<= {val}"
                else:
                    line += f"{val}x{idx+1}"

            line += " <= "
            line += self.tableau_b[-1]
        print(line)

    def add_cut(self, a, b):
        '''
        Add new constraint to the problem
        '''
        pass

    def ppt(self):
        print('const:', self.tableau_const)
        print('c:\n',   self.c_tableau)
        print('A:\n',   self.A_tableau)
        print('b:\n',   self.b_tableau)
        print('======================')


if __name__ == '__main__':
    A = np.array([
        [1, 0, 0],
        [2, 1, 1],
        [2, 2, 1],
    ])
    b = np.array([4, 10, 16])
    c = np.array([20, 16, 12])
    const = -10
    simplex = Simplex(A, b, c, const)
    simplex.solve()
    #simplex.pivot()
    #simplex.ppt()
    #simplex.pivot()
    #simplex.ppt()
