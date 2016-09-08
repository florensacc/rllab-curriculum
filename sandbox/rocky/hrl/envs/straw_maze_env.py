

from rllab.envs.base import Env, Step
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
import numpy as np
import random


class STRAWMazeEnv(Env):
    def __init__(self, r=0.05):
        self.map = np.array(list(map(list, [
            "xxxxxxxxxxx",
            "x.x.......x",
            "x.x.xxx.x.x",
            "x...x..xx.x",
            "xxxxx.xxx.x",
            "x.....x...x",
            "x.xxxxxxxxx",
            "x.x.......x",
            "x.xxxxxxx.x",
            "x.........x",
            "xxxxxxxxxxx",
        ])))
        self.free_positions = np.array(list(zip(*list(map(list, np.where(self.map == '.'))))))
        self.agent_pos = None
        self.goal_pos = None
        self.obs_template = np.eye(4)[np.cast['int'](self.map == 'x')]
        self.r = r
        self.reset()

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(11, 11, 4))

    @property
    def action_space(self):
        return Discrete(4)

    def reset(self):
        self.agent_pos, self.goal_pos = random.sample(self.free_positions, k=2)
        return self.get_current_obs()

    def get_current_obs(self):
        # 0: free space
        # 1: obstacle
        # 2: agent
        # 3: goal
        obs = np.copy(self.obs_template)
        obs[tuple(self.agent_pos) + (2,)] = 1
        obs[tuple(self.goal_pos) + (3,)] = 1
        return obs

    def step(self, action):
        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        next_pos = self.agent_pos + increments[action]
        done = False
        reward = 0
        if self.map[tuple(next_pos)] == 'x':
            reward = -2 * self.r
        elif tuple(next_pos) == tuple(self.goal_pos):
            reward = self.r
            done = True
            self.agent_pos = next_pos
        else:
            reward = -self.r
            self.agent_pos = next_pos
        return Step(self.get_current_obs(), reward, done)


if __name__ == "__main__":
    env = STRAWMazeEnv()
    obs = env.reset()
    env.step(1)
    import ipdb;

    ipdb.set_trace()
