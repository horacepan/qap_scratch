import pdb
import scipy.io

import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim

def sinkhorn(X, iters=100):
    ones = torch.ones(X.shape[0])
    for i in range(iters):
        X /= X.sum(dim=0, keepdim=True)
        X /= X.sum(dim=1, keepdim=True)

        if torch.allclose(X.sum(dim=0), ones) and torch.allclose(X.sum(dim=1), ones):
            #print(f'Sinkhorn okay in {i} iters')
            return X

    return X

class QAPnet(nn.Module):
    def __init__(self, F, D):
        super(QAPnet, self).__init__()
        self.F = torch.from_numpy(F).float()
        self.D = torch.from_numpy(D).float()
        self.X = nn.Parameter(sinkhorn(torch.abs(torch.rand(F.shape))))

    def forward(self):
        perm_D = self.X.mm(self.D).mm(self.X.t())
        return (self.F * perm_D).sum()

    def birkhoff_projection(self):
        with torch.no_grad():
            self.X.data = sinkhorn(self.X.data, 200)

    def eval(self):
        with torch.no_grad():
            perm_D = self.X.mm(self.D).mm(self.X.t())
            return (self.F * perm_D).sum()

def solve_qap(F, D):
    lr = 0.0001
    epochs = 1000
    sinkhorn_steps = 10
    logiters = 100

    net = QAPnet(F, D)
    opt = optim.Adam(net.parameters(), lr=lr)

    for e in range(epochs):
        for _ in range(sinkhorn_steps):
            loss = net.forward()
            opt.zero_grad()
            loss.backward()
            opt.step()

        net.birkhoff_projection()

        if e % logiters == 0:
            print(f'Epoch {e:5d} | Loss: {loss.item():.2f}')

    net.birkhoff_projection()
    print(f'Final loss: {net().item()}')
    pdb.set_trace()

if __name__ == '__main__':
    mat = scipy.io.loadmat('data/nug12.mat')
    F = mat['A']
    D = mat['B']
    solve_qap(F, D)
