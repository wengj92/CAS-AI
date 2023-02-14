from collections import deque
import numpy as np


class RewardTracker:
    def __init__(self):
        self.rewards_100 = deque(maxlen=100)

    def __call__(self, rewards):
        mean_reward = np.mean(rewards)
        # append to rewards_100
        self.rewards_100.append(mean_reward)
        mean_reward_100 = np.mean(self.rewards_100)
        return mean_reward, mean_reward_100


