from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pygame
import random
from rllab.envs.box2d.parser import find_body
from rllab.envs.base import Step

from rllab.core.serializable import Serializable
from rllab.envs.box2d.box2d_env import Box2DEnv


class BimountainCarEnv(Box2DEnv, Serializable):
    def __init__(self,
                 height_bonus=1.,
                 goal_cart_pos=0.6,
                 *args, **kwargs):
        super(BimountainCarEnv, self).__init__(
            self.model_path("mountain_car.xml.mako"),
            *args, **kwargs
        )
        self.max_cart_pos = 2
        self.goal_cart_pos = goal_cart_pos
        self.height_bonus = height_bonus
        self.cart = find_body(self.world, "cart")
        self.reset_trial()
        Serializable.quick_init(self, locals())

    def reset_trial(self):
        self.goal_cart_pos = self.goal_cart_pos * random.choice([-1, 1])
        return self.reset()

    def compute_reward(self, action):
        yield
        yield (-1 + self.height_bonus * self.cart.position[1])

    def is_current_done(self):
        if self.goal_cart_pos >= 0:
            return self.cart.position[0] >= self.goal_cart_pos \
                   or abs(self.cart.position[0]) >= self.max_cart_pos
        else:
            return self.cart.position[0] < self.goal_cart_pos \
                   or abs(self.cart.position[0]) >= self.max_cart_pos

    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        bounds = np.array([
            [-1],
            [1],
        ])
        low, high = bounds
        xvel = np.random.uniform(low, high)
        self.cart.linearVelocity = (xvel, self.cart.linearVelocity[1])
        return self.get_current_obs()

    def step(self, action):
        next_obs, reward, done, info = super(BimountainCarEnv, self).step(action)
        return Step(next_obs, reward, done, **dict(info, goal_cart_pos=self.goal_cart_pos))

    def action_from_keys(self, keys):
        if keys[pygame.K_LEFT]:
            return np.asarray([-1])
        elif keys[pygame.K_RIGHT]:
            return np.asarray([+1])
        else:
            return np.asarray([0])
