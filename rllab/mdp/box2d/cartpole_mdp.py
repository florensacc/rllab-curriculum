from rllab.mdp.box2d.box2d_mdp import Box2DMDP
from rllab.mdp.box2d.parser import find_body
import numpy as np
from rllab.core.serializable import Serializable


class CartpoleMDP(Box2DMDP, Serializable):

    def __init__(self):
        super(CartpoleMDP, self).__init__(self.model_path("cartpole.xml"))
        self.max_pole_angle = .2
        self.max_cart_pos = 2.4
        self.cart = find_body(self.world, "cart")
        self.pole = find_body(self.world, "pole")
        Serializable.__init__(self)

    def get_current_reward(self, action):
        notdone = 1 - int(self.is_current_done())
        ucost = 1e-5*(action**2).sum()
        xcost = 1 - np.cos(self.pole.angle)
        return notdone * 10 - notdone * xcost - notdone * ucost

    def is_current_done(self):
        return abs(self.cart.position[0]) > self.max_cart_pos or \
            abs(self.pole.angle) > self.max_pole_angle
