import pygame

from rllab.mdp.box2d.box2d_mdp import Box2DMDP
from rllab.mdp.box2d.box2d_viewer import Box2DViewer
from rllab.mdp.box2d.parser import find_body
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


# Tornio, Matti, and Tapani Raiko. "Variational Bayesian approach for nonlinear identification and control." Proc. of the IFAC Workshop on Nonlinear Model Predictive Control for Fast Systems, NMPC FS06. 2006.
class CartpoleSwingupMDP(Box2DMDP, Serializable):

    @autoargs.inherit(Box2DMDP.__init__)
    def __init__(self, **kwargs):
        super(CartpoleSwingupMDP, self).__init__(
            self.model_path("cartpole.xml"),
            **kwargs
        )
        self.max_cart_pos = 3
        self.max_reward_cart_pos = 3
        self.cart = find_body(self.world, "cart")
        self.pole = find_body(self.world, "pole")
        Serializable.__init__(self)

    @overrides
    def reset(self):
        self.set_state(self.initial_state)
        bounds = np.array([
            [-1, -2, np.pi-1, -3],
            [1, 2, np.pi+1, 3],
        ])
        low, high = bounds
        xpos, xvel, apos, avel = np.random.uniform(low, high)
        self.cart.position = (xpos, self.cart.position[1])
        self.cart.linearVelocity = (xvel, self.cart.linearVelocity[1])
        self.pole.angle = apos
        self.pole.angularVelocity = avel
        return self.get_state(), self.get_current_obs()

    @overrides
    def get_current_reward(self, action):
        if self.is_current_done():
            return -100
        else:
            if abs(self.cart.position[0]) > self.max_reward_cart_pos:
                return -1
            else:
                return np.cos(self.pole.angle)

    @overrides
    def is_current_done(self):
        return abs(self.cart.position[0]) > self.max_cart_pos

    @overrides
    def action_from_keys(self, keys):
        if keys[pygame.K_LEFT]:
            return np.asarray([-10])
        elif keys[pygame.K_RIGHT]:
            return np.asarray([+10])
        else:
            return np.asarray([0])
