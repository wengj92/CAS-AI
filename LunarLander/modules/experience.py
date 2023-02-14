from collections import deque
import numpy as np
import gym


class ExperienceSource:
    def __init__(self, env, agent, gamma):
        self.env = env
        self.agent = agent
        self.gamma = gamma

    def _reset_env(self):
        self.states = np.array([self.env.reset()[0]])
        self.rewards = np.array([])
        self.actions = np.array([])

    def __iter__(self):
        # init variables
        self._reset_env()
        # loop until program finishes
        while True:
            # get action from agent
            action = self.agent(self.states[-1]).item()
            # step environment
            next_state, reward, finished, _, _ = self.env.step(action)
            # append to list
            self.states = np.insert(self.states, 1, next_state, axis=0)
            self.actions = np.append(self.actions, action)
            self.rewards = np.append(self.rewards, reward)
            # check if episode is finished
            if finished:
                # get discounted rewards
                self.total_reward = 0.0
                for reward in reversed(self.rewards):
                    self.total_reward *= self.gamma
                    self.total_reward += reward
                # return results and continue loop
                yield self.states[:-1], self.actions, self.rewards
                # now, reset the environment
                self._reset_env()

    def unpack_batch(self, batch):
        states = batch[0][0]
        actions = batch[0][1]
        rewards = batch[0][2]
        for episode in batch[1:]:
            states = np.insert(states, 1, episode[0], axis=0)
            actions = np.append(actions, episode[1])
            rewards = np.append(rewards, episode[2])
        return states, actions, rewards

    def get_reward(self):
        return self.total_reward


class ExperienceSourceFirstLast:
    def __init__(self, environment_name, number_environments, agent, steps, gamma):
        self.environments = [gym.make(environment_name) for _ in range(number_environments)]
        self.agent = agent
        self.steps = steps
        self.gamma = gamma
        # init class variables
        self.rewards = []

    def _get_experience(self):
        # init variables
        states = []
        history = []
        total_rewards = []
        # reset all environments and append to states list
        for i, environments in enumerate(self.environments):
            # reset the environment
            states.append(environments.reset()[0])
            history.append(deque(maxlen=self.steps))
            total_rewards.append(0.0)
        # iterate until code completes
        while True:
            # get next actions and predictions
            actions = self.agent(states)
            # iterate through environments
            for i, (environment, state, action) in enumerate(zip(self.environments, states, actions)):
                # step environment
                next_state, reward, finished, _, _ = environment.step(action.item())
                # append values to history
                history[i].append([state, action, reward, finished])
                # calculate total rewards
                total_rewards[i] += reward
                # check if finished and reset the environment if yes
                if finished:
                    # return tail of the history
                    while len(history[i]) >= 1:
                        yield history[i]
                        history[i].popleft()
                    # store total reward and reset reward for the environment to zero
                    self.rewards.append(total_rewards[i])
                    total_rewards[i] = 0.0
                    # reset environment
                    next_state = environment.reset()[0]
                    # clear history of corresponding environment
                    history[i].clear()
                # check if number of steps required are done
                if len(history[i]) == self.steps:
                    # return the history of the last steps
                    yield history[i]
                # state for next iteration of the environment is currently named next ste
                states[i] = next_state

    def __iter__(self):
        # experiences is a list with [state, action, reward, finished]
        for experiences in self._get_experience():
            state = experiences[0][0]
            last_state = experiences[-1][0]
            action = experiences[0][1]
            is_done = experiences[-1][3]
            # calculate discounted rewards
            reward = 0.0
            for experience in reversed(experiences):
                reward *= self.gamma
                reward += experience[2]
            # return state, action, reward, last state and done
            yield state, action, reward, last_state, is_done

    def unpack_batch(self, batch):
        states = np.array([batch[0][0]])
        actions = batch[0][1]
        rewards = batch[0][2]
        for episode in batch[1:]:
            states = np.insert(states, 1, episode[0], axis=0)
            actions = np.append(actions, episode[1])
            rewards = np.append(rewards, episode[2])
        return states, actions, rewards

    def get_reward(self):
        # store copy and reset list to empty
        rewards = self.rewards
        if rewards:
            self.rewards = []
        # return copy of list with rewards
        return rewards

