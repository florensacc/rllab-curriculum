from __future__ import print_function
from __future__ import absolute_import
from sandbox.rocky.hrl.envs.supervised_env import SupervisedEnv
from sandbox.rocky.hrl.misc.hrl_utils import using_seed
from rllab.envs.base import Step
from sandbox.rocky.hrl.envs.env_util import GridPlot
from rllab.spaces.product import Product
from rllab.spaces.discrete import Discrete
from rllab.core.serializable import Serializable
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
import sys


class PermGridEnv(SupervisedEnv, Serializable):
    def __init__(self, size=5, n_objects=5, order_length=None, object_seed=None, perm_seed=None, n_fixed_perm=None,
                 random_restart=True):
        """
        :param size: Size of the grid world
        :param n_objects: Number of objects in the grid world
        :param order_length: Length of the instruction. By default it is equal to the number of objects
        :param object_seed: Seed used to sample object positions. Setting the seed will ensure consistent object
        positions.
        :param perm_seed: Seed used to sample permutations. Only used if fixed_perm is True.
        :param n_fixed_perm: number of fixed permutations to use. If set to None (default), each time a completely
        random permutation will be used
        :param random_restart: Whether to randomly sample the agent's position every episode. If false, agent will
        always start from the top left corner.
        """
        if object_seed is None:
            object_seed = np.random.randint(sys.maxint)
        if perm_seed is None:
            perm_seed = np.random.randint(sys.maxint)
        Serializable.quick_init(self, locals())
        self.size = size
        self.n_objects = n_objects
        if order_length is None:
            order_length = n_objects
        assert order_length <= n_objects, "Cannot have order length greater than the number of objects"
        # initialize object positions
        with using_seed(object_seed):
            # ensure objects are all in different positions
            self.object_positions = [
                (x / self.size, x % self.size) for x in np.random.permutation(self.size * self.size)[:self.n_objects]
                ]
        if n_fixed_perm is not None:
            assert n_fixed_perm >= 1
            with using_seed(perm_seed):
                self.perms = [tuple(np.random.permutation(self.n_objects)[:order_length]) for _ in xrange(n_fixed_perm)]
        else:
            self.perms = None
        self.n_fixed_perm = n_fixed_perm
        self.order_length = order_length
        self.fig = None
        self.agent_pos = None
        self.visit_order = None
        self.n_visited = None
        self.random_restart = random_restart
        self._observation_space = Product([
            Product([Discrete(self.size), Discrete(self.size)]),
            Product([Discrete(self.n_objects) for _ in xrange(self.order_length)]),
            Product([Discrete(2) for _ in xrange(self.order_length)]),
        ])
        self._action_space = Discrete(4)
        self.reset()

    def reset(self):
        if self.random_restart:
            self.agent_pos = self.object_positions[0]
            while self.agent_pos in self.object_positions:
                self.agent_pos = tuple(np.random.randint(low=0, high=self.size, size=(2,)))
        else:
            self.agent_pos = (0, 0)
        if self.n_fixed_perm is not None:
            assert self.perms is not None
            self.visit_order = random.choice(self.perms)
        else:
            self.visit_order = tuple(np.random.permutation(self.n_objects))[:self.order_length]
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
            tuple([1] * self.n_visited + [0] * (self.order_length - self.n_visited)),
        )

    def step(self, action):
        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        self.agent_pos = tuple(np.clip(
            np.array(self.agent_pos) + increments[action],
            [0, 0],
            [self.size - 1, self.size - 1]
        ))
        if self.object_positions[self.visit_order[self.n_visited]] == self.agent_pos:
            self.n_visited += 1
            reward = 1
        else:
            reward = 0
        done = self.n_visited == self.order_length
        return Step(observation=self.get_current_obs(), reward=reward, done=done)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

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
