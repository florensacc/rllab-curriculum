from __future__ import print_function
from __future__ import absolute_import
from rllab.envs.base import Env
from rllab.envs.base import Step
from sandbox.rocky.hrl.envs.env_util import GridPlot
from rllab.spaces.product import Product
from rllab.spaces.discrete import Discrete
import matplotlib.pyplot as plt
import numpy as np


class PermGridEnv(Env):
    def __init__(self, size=5, n_objects=5, object_seed=None):
        self.size = size
        # initialize object positions
        rng_state = None
        if object_seed:
            rng_state = np.random.get_state()
            np.random.seed(object_seed)
        # ensure objects are all in different positions
        self.n_objects = n_objects
        self.object_positions = [
            (x / self.size, x % self.size) for x in np.random.permutation(self.size * self.size)[:self.n_objects]
        ]
        # self.object_positions = np.random.randint(low=0, high=size, size=(n_objects, 2))
        if object_seed:
            np.random.set_state(rng_state)
        self.fig = None
        self.agent_pos = None
        self.visit_order = None
        self.n_visited = 0

    def reset(self):
        self.agent_pos = self.object_positions[0]
        while self.agent_pos in self.object_positions:
            self.agent_pos = tuple(np.random.randint(low=0, high=self.size, size=(2,)))
        self.visit_order = tuple(np.random.permutation(self.n_objects))
        self.n_visited = 0
        return self.get_current_obs()

    @staticmethod
    def action_from_direction(d):
        """
        Return the action corresponding to the given direction. This is a helper method for debugging and testing
        purposes.
        :return: the action index corresponding to the given direction
        """
        return dict(
            left=0,
            down=1,
            right=2,
            up=3
        )[d]

    def get_current_obs(self):
        # return: current position, order to visit objects, and whether each objects have been visited
        return (
            self.agent_pos,
            self.visit_order,
            tuple([1] * self.n_visited + [0] * (self.n_objects - self.n_visited)),
        )

    def step(self, action):
        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        self.agent_pos = tuple(np.clip(
            self.agent_pos + increments[action],
            [0, 0],
            [self.size - 1, self.size - 1]
        ))
        if self.object_positions[self.visit_order[self.n_visited]] == self.agent_pos:
            self.n_visited += 1
            reward = 1
        else:
            reward = 0
        done = self.n_visited == self.n_objects
        return Step(observation=self.get_current_obs(), reward=reward, done=done)

    @property
    def observation_space(self):
        return Product([
            Product([Discrete(self.size), Discrete(self.size)]),
            Product([Discrete(self.n_objects) for _ in xrange(self.n_objects)]),
            Product([Discrete(2) for _ in xrange(self.n_objects)]),
        ])

    @property
    def action_space(self):
        return Discrete(4)

    def render(self):
        if self.fig is None:
            self.fig = GridPlot(self.size)
            plt.ion()
        self.fig.reset_grid()
        self.fig.color_grid(self.agent_pos[0], self.agent_pos[1], 'g')
        self.fig.add_text(self.agent_pos[0], self.agent_pos[1], 'Agent')
        for idx, visit_idx in enumerate(self.visit_order):
            pos = self.object_positions[visit_idx]
            if idx >= self.n_visited:
                self.fig.color_grid(pos[0], pos[1], 'b')
                self.fig.add_text(pos[0], pos[1], 'Obj #%d' % idx)
        plt.pause(0.001)
