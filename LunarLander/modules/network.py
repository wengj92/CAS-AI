import numpy as np

import torch
import torch.nn as nn
import torch.distributions as distributions


class ActorCriticNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_hidden=128):
        super(ActorCriticNet, self).__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
            nn.ReLU()
        )
        # actor net
        self.actor = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, output_dim),
            nn.LogSoftmax(dim=-1)
        )
        # critic net
        self.critic = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x):
        x = self.input(x)
        return self.actor(x), self.critic(x)


class ActorCriticAgent:
    def __init__(self, net):
        self.net = net

    def _preprocessor(self, states):
        if len(states) == 1:
            return torch.Tensor(np.array([states]))
        else:
            return torch.Tensor(np.array([s for s in states]))

    def __call__(self, states):
        # create tensor from list of arrays
        states = self._preprocessor(states)
        # calculate network output
        log_probablilites, _ = self.net.forward(states)
        # get next action based on probability distribution
        action = distributions.Categorical(log_probablilites).sample()
        # return action, action log probability and value prediction
        return action

