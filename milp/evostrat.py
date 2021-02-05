import pdb
import time
from functools import partial
import multiprocessing as mp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import gym

def get_reward_cp(env, model, weights):
    for idx, w in enumerate(model.parameters()):
        w.data = weights[idx]

    state = env.reset()
    state = torch.FloatTensor(state)
    tot_reward = 0
    done = False
    while not done:
        action_probs = F.softmax(model.forward(state), dim=0)
        distr = Categorical(action_probs)
        action = distr.sample().item()

        state, reward, done, _ = env.step(action)
        state = torch.FloatTensor(state)
        tot_reward += reward

    return tot_reward

def add_weight(weights, p):
    return [weights[i] + p[i] for i in range(len(weights))]

def worker_process(args):
    reward_func, weights = args
    reward = reward_func(weights)
    return reward

class EvoStrat:
    def __init__(self, weights, reward_func, pop_size, mu, sigma, learning_rate, ncpu=-1, normalize_rewards=False):
        self.weights = weights
        self.get_reward = reward_func
        self.pop_size = pop_size
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.ncpu = mp.cpu_count() if ncpu == -1 else ncpu
        self.normalize_rewards = normalize_rewards

    def get_rewards(self, pool, pop_weights):
        '''
        Returns: list of floats
        '''
        if pool is not None:
            # length should be equal to size of population
            worker_args = [(self.get_reward, add_weight(self.weights, p)) for p in pop_weights]
            rewards = pool.map(worker_process, worker_args)
        else:
            rewards = []
            for p in pop_weights:
                w = add_weight(self.weights, p)
                r = self.get_reward(w)
                rewards.append(r)

        rewards = np.array(rewards, dtype=np.float32)
        return rewards

    def update_weights(self, rewards, pop_weights):
        '''
        rewards: list of floats
        weights: list of weights (list of numpy matrices)
        '''
        grad_weights = []

        if self.normalize_rewards:
            rewards = (rewards - rewards.mean()) / rewards.std()

        for idx in range(len(self.weights)):
            # get the population weights of this
            pop_layer = np.array([p[idx] for p in pop_weights])
            coeff = self.learning_rate / (self.sigma * self.pop_size)
            self.weights[idx] += coeff * np.dot(pop_layer.T, rewards).T

    def get_pop_weights(self):
        '''
        Returns: list of weights (list of numpy matrices)
        '''
        pop_weights = []
        for p in range(self.pop_size):
            x = []
            for idx, w in enumerate(self.weights):
                eps = np.random.normal(size=w.shape).astype(np.float32)
                x.append(eps)

            pop_weights.append(x)

        return pop_weights

    def get_weights(self):
        '''
        Returns: list of weights (list of numpy matrices)
        '''
        return self.weights

    def run(self, niters, log_iters=100):
        '''
        niters: int, number of gradient steps to take
        log_iters: int, number of iterations between printing progress
        '''
        st = time.time()
        avg_rewards = []
        pool = mp.Pool(self.ncpu) if self.ncpu > 1 else None
        for i in range(niters):
            pop_weights = self.get_pop_weights()
            rewards = self.get_rewards(pool, pop_weights) # where the magic happens
            self.update_weights(rewards, pop_weights)
            avg_rewards.append(np.mean(rewards))

            if i % log_iters == 0:
                print("Iter: {:3d} | Last 100 avg reward: {:.2f} | Rollout: {:.2f} | Elapsed: {:.2f}min".format(
                    i, np.mean(avg_rewards[-100:]), self.get_reward(self.get_weights()), (time.time() - st) / 60.
                ))

        if pool is not None:
            pool.close()
            pool.join()

if __name__ == '__main__':
    in_dim = 4
    hid_dim = 32
    out_dim = 2

    pop_size = 20
    mu = 0
    sigma = 0.1
    learning_rate = 0.01
    ncpu = -1
    normalize_rewards = 0
    env = gym.make("CartPole-v0")

    model = nn.Sequential(
        nn.Linear(in_dim, hid_dim),
        nn.ReLU(),
        nn.Linear(hid_dim, out_dim)
    )
    model = model.float()

    weights = [p.data for p in model.parameters()]
    reward_func = partial(get_reward_cp, env, model)
    evo = EvoStrat(weights, reward_func, pop_size, mu, sigma, learning_rate, ncpu=ncpu, normalize_rewards=False)
    evo.run(1000)
