import pygame

from rllab.mdp.box2d.box2d_mdp import Box2DMDP
from rllab.mdp.box2d.box2d_viewer import Box2DViewer
from rllab.mdp.box2d.parser import find_body
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class MountainCarMDP(Box2DMDP, Serializable):

    @autoargs.inherit(Box2DMDP.__init__)
    def __init__(self, **kwargs):
        super(MountainCarMDP, self).__init__(
            self.model_path("mountain_car.xml"),
            **kwargs
        )
        self.goal_cart_pos = 0.6
        self.max_cart_pos = 2
        self.cart = find_body(self.world, "cart")
        Serializable.__init__(self)

    @overrides
    def get_current_reward(
            self, state, raw_obs, action, next_state, next_raw_obs):
        return -1 + self.cart.position[1]

    @overrides
    def is_current_done(self):
        return self.cart.position[0] >= self.goal_cart_pos \
               or abs(self.cart.position[0]) >= self.max_cart_pos

    @overrides
    def reset(self):
        self._set_state(self.initial_state)
        bounds = np.array([
            [-1],
            [1],
        ])
        low, high = bounds
        xvel = np.random.uniform(low, high)
        self.cart.linearVelocity = (xvel, self.cart.linearVelocity[1])
        return self.get_state(), self.get_current_obs()

    @overrides
    def action_from_keys(self, keys):
        if keys[pygame.K_LEFT]:
            return np.asarray([-1])
        elif keys[pygame.K_RIGHT]:
            return np.asarray([+1])
        else:
            return np.asarray([0])
