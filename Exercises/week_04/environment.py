import numpy as np
import random
import gym
from gym import spaces
import pygame


class TicTacToe(gym.Env):
    def __init__(self, size):
        # set size of tic tac toe board
        self.size = size
        self.board_size = [size, size]
        # initialize board
        self.board = np.zeros(self.board_size, dtype=int)
        # initialize action space
        self.action_space = spaces.Discrete(size * size)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(size, size), dtype=int)
        # init window
        self.window = None

    def _action_2d(self, action):
        x = action % self.size
        y = int(action / self.size)
        return [x, y]

    def _check_if_possible(self, action):
        if self.board[action[1], action[0]] == 0:
            # return state and reward
            return True, 0
        else:
            # return state and reward
            return False, -1

    def _check_game_over(self, winner):
        # check rows and columns
        for i in range(self.size):
            if np.abs(np.sum(self.board[i,:])) == self.size or np.abs(np.sum(self.board[:,i])) == self.size:
                # return done and reward
                return True, winner * 100
        # check diagonals
        diag1 = np.abs(np.sum([self.board[i, i] for i in range(self.size)]))
        diag2 = np.abs(np.sum([self.board[i, -(i+1)] for i in range(self.size)]))
        if diag1 == self.size or diag2 == self.size:
            # return done and reward
            return True, winner * 100
        # check if board is full
        if not np.any(self.board==0):
            # return done and reward
            return True, 0
        # otherwise, continue playing
        return False, 0
        
    def step(self, action):
        # convert action to 2d
        action = self._action_2d(action)
        # check if action is possible
        is_possible, reward = self._check_if_possible(action)
        if is_possible:
            # if the action is possible, set id of the corresponding field to 1
            self.board[action[1], action[0]] = 1
            # check if the game is over
            done, reward = self._check_game_over(1)
            if done:
                return self.board, reward, True, False, {}
            # otherwise, continue playing
            else:
                # get possible choices
                options = np.argwhere(self.board == 0)
                # sample random action from environment
                action = random.choice(options)
                # set id of the corresponding field to -1
                self.board[action[0], action[1]] = -1
                # check if game is over
                done, reward = self._check_game_over(-1)
                if done:
                    return self.board, reward, True, False, {}
        # punish if action is not possible and have the agent play again
        else:
            return self.board, reward, False, False, {}
        # if nothing else happened and the game isn't over yet, return 0 as reward
        return self.board, 0, False, False, {}      
        
    def reset(self, seed=None, options=None):
        # use reset method of parent class using super function call
        super().reset(seed=seed)
        # reset board
        self.board = np.zeros(self.board_size, dtype=int)
        # randomly have the opponent start a game
        if random.choice([True, False]):
            action = self._action_2d(self.action_space.sample())
            self.board[action[1], action[0]] = -1
        # return observations and info dict
        return self.board, {}

    def render(self):
        # init window
        if self.window is None:
            self.window_size = 512
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        # define canvas (white fill)
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        # size of a single grid square in pixels
        square_size = (self.window_size / self.size)
        # draw dots
        nr_of_fileds = self.size * self.size
        for x in range(self.size):
            for y in range(self.size):
                current_location = np.array([x, y])
                # get current location
                if self.board[x, y] == 1:
                    pygame.draw.circle(canvas, (0, 0, 255), (current_location + 0.5) * square_size, square_size / self.size,)
                elif self.board[x, y] == -1:
                    pygame.draw.circle(canvas, (0, 255, 0), (current_location + 0.5) * square_size, square_size / self.size,)
                else:
                    pass
        # add gridlines to canvas
        for i in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, square_size * i), (self.window_size, square_size * i), width=3,)
            pygame.draw.line(canvas, 0, (square_size * i, 0), (square_size * i, self.window_size), width=3,)
        # The following line copies our drawings from canvas to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update() 

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# EOF
