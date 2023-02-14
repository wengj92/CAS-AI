import numpy as np
import gym
import torch.optim as optim
import torch.nn.utils as nn_utils

# import custom modules
from modules.network import ActorCriticNet, ActorCriticAgent
from modules.experience import ExperienceSource, ExperienceSourceFirstLast
from modules.loss import calculate_losses
from modules.tracker import RewardTracker

# environment settings
environment_name = 'CartPole-v0'
number_environments = 1
reward_threshold = 250

# optimizer settings
learning_rate = 0.0001
batch_size = 128

# training parameters
training_steps = 5
gamma = 0.99
clip_grad = 1e-3

# create environments and get state and action space
env = gym.make(environment_name)
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

# create network and ageng
net = ActorCriticNet(state_space, action_space)
agent = ActorCriticAgent(net)

# define optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# initialize experience source
if number_environments > 1:
    experience_source = ExperienceSource(environment_name, number_environments, agent, training_steps, gamma)
else:
    experience_source = ExperienceSource(env, agent, 0.99)

# create instance of reward tracker
reward_tracker = RewardTracker()

# loop until reward is greater than threshold
batch = []
for i, experience in enumerate(experience_source):
    # create batch
    batch.append(experience)

    # check if rewards threshold is reached
    rewards = experience_source.get_reward()
    if rewards:
        current_reward, mean_reward = reward_tracker(rewards)
        if np.mean(mean_reward) >= reward_threshold:
            print(f'Training finished {i} iterations with mean reward={mean_reward}')
            break

    # reiterate until length of observations equals batch_size parameter
    if len(batch) < batch_size:
        continue

    # print debug message
    if rewards:
        print(f'%i iterations:\t mean reward: %0.3f' % (int((i + 1) / batch_size), mean_reward))

    # unpack batch
    batch = experience_source.unpack_batch(batch)

    # calculate losses
    critic_loss, actor_loss = calculate_losses(net, batch)

    # set gradients to zero
    optimizer.zero_grad()

    # backward pass
    actor_loss.backward(retain_graph=True)
    critic_loss.backward()

    # clip gradients
    nn_utils.clip_grad_norm_(net.parameters(), clip_grad)

    # step both optimizers
    optimizer.step()

    # clear batch
    batch = []


