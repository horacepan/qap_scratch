import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from milp_env import MaxCutMILP

def make_mlp(in_dim, hid_dim, out_dim, nlayers=1):
    layers = []
    layers.append(nn.Linear(in_dim, hid_dim))
    layers.append(nn.ReLU())

    for _ in range(nlayers):
        layers.append(nn.Linear(hid_dim, hid_dim))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(hid_dim, out_dim))
    return nn.Sequential(*layers)

class MLPAttentionPolicy(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, nlayers, normalize=True):
        super(MLPAttentionPolicy, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.normalize = normalize
        self.mlp_embed = make_mlp(in_dim, hid_dim, out_dim, nlayers)

    def act(self, obs):
        A, b, c0, curr_sol, (A_cuts, b_cuts) = obs
        A = torch.from_numpy(A).float()
        b = torch.from_numpy(b).unsqueeze(-1).float()
        A_cuts = torch.from_numpy(A_cuts).float()
        b_cuts = torch.from_numpy(b_cuts).unsqueeze(-1).float()

        Ab = torch.cat([A, b], dim=1)
        cut_ab = torch.cat([A_cuts, b_cuts], dim=1)
        all_ob = torch.cat([Ab, cut_ab], dim=0)

        if self.normalize:
            all_ob = (all_ob - all_ob.mean()) / (all_ob.max() - all_ob.min() + 1e-8)

        constraints = all_ob[:A.shape[0], :]
        cuts = all_ob[A.shape[0]:, :]
        constraints_embed = self.mlp_embed(constraints)
        cuts_embed = self.mlp_embed(cuts)

        att_map = cuts_embed.matmul(constraints_embed.T)
        score = att_map.mean(dim=1)
        score -= score.max()
        probs = F.softmax(score, dim=0)
        action = Categorical(probs).sample()
        return action.item()

    def forward(self, state):
        return self.act(state)

if __name__ == '__main__':
    num_verts = 7
    num_edges = 13
    in_dim = num_verts + num_edges + 1
    hid_dim = 64
    out_dim = 32
    nlayers = 2

    env = MaxCutMILP(num_verts, num_edges)
    m = MLPAttentionPolicy(in_dim, hid_dim, out_dim, nlayers)
    state = env.reset()
    print(m(state))
