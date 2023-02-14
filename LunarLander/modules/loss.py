import numpy as np
import torch


def calculate_losses(net, batch, device='cpu'):
    # initialize variables
    states, actions, rewards = batch
    # convert lists to torch tensors
    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    # get values from net
    log_probabilites, values = net(states)
    # calculate advantages values
    advantages = rewards - values.flatten()
    # calculate critic loss
    critic_loss = advantages.pow(2)
    # calculate actor loss
    actor_loss = -log_probabilites[:, actions] * advantages.detach()
    # return tensors
    return critic_loss.mean(), actor_loss.mean()