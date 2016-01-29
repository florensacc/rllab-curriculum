import pygame

from rllab.mdp.box2d.box2d_mdp import Box2DMDP
from rllab.mdp.box2d.parser import find_body, find_joint
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class InvertedDoublePendulumMDP(Box2DMDP, Serializable):

    @autoargs.inherit(Box2DMDP.__init__)
    def __init__(self, *args, **kwargs):
        kwargs["frame_skip"] = kwargs.get("frame_skip", 20)
        super(InvertedDoublePendulumMDP, self).__init__(
            self.model_path("inverted_double_pendulum.xml"),
            *args, **kwargs
        )
        self.link_len = 1
        self.cart = find_body(self.world, "cart")
        self.link1 = find_body(self.world, "link1")
        self.link2 = find_body(self.world, "link2")
        self.tgt_tip_pos = self.get_tip_pos()
        Serializable.__init__(self, *args, **kwargs)

    @overrides
    def reset(self):
        self._set_state(self.initial_state)
        # stds = np.array([0.1, 0.1, 0.01, 0.01])
        # pos1, pos2, v1, v2 = np.random.randn(*stds.shape) * stds
        # self.link1.angle = pos1
        # self.link2.angle = pos2
        # self.link1.angularVelocity = vu
        # self.link2.angularVelocity = v2
        return self.get_state(), self.get_current_obs()

    def get_tip_pos(self):
        cur_center_pos = self.link2.position
        cur_angle = self.link2.angle
        cur_pos = (
            cur_center_pos[0] + self.link_len*np.sin(cur_angle),
            cur_center_pos[1] + self.link_len*np.cos(cur_angle)
        )
        return cur_pos

    @overrides
    def compute_reward(self, action):
        yield
        cur_pos = self.get_tip_pos()
        yield -(0.1 * (cur_pos[0] - self.tgt_tip_pos[0])**2 + (cur_pos[1] - self.tgt_tip_pos[1])**2)

    def is_current_done(self):
        return False

    @overrides
    def action_from_keys(self, keys):
        if keys[pygame.K_LEFT]:
            return np.asarray([-10])
        elif keys[pygame.K_RIGHT]:
            return np.asarray([+10])
        else:
            return np.asarray([0])

