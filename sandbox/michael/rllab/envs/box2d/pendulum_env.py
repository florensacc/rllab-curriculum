import numpy as np
import math
from rllab.envs.box2d.parser import find_body

from rllab.core.serializable import Serializable
from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


# http://mlg.eng.cam.ac.uk/pilco/
class PendulumEnv(Box2DEnv, Serializable):
    @autoargs.inherit(Box2DEnv.__init__)
    def __init__(self, eps_sparse=None, *args, **kwargs):
        self.eps_sparse = eps_sparse  # how far from goal we receive a reward of 1
        # make sure mdp-level step is 100ms long
        kwargs["frame_skip"] = kwargs.get("frame_skip", 2)
        if kwargs.get("template_args", {}).get("noise", False):
            self.link_len = (np.random.rand() - 0.5) + 1
        else:
            self.link_len = 1
        kwargs["template_args"] = kwargs.get("template_args", {})
        kwargs["template_args"]["link_len"] = self.link_len
        super(PendulumEnv, self).__init__(
            self.model_path("pendulum.xml.mako"),
            *args, **kwargs
        )
        self.link1 = find_body(self.world, "link1")
        Serializable.__init__(self, *args, **kwargs)

    @overrides
    def reset(self, noisy=False, initial_state=None):
        """Assumes state give in same format as get_obs: sin/cos/aVel"""
        self._invalidate_state_caches()
        self._set_state(self.initial_state)
        if initial_state is not None:
            initial_angle = math.atan2(initial_state[0], initial_state[1])
            self.link1.angle = initial_angle
            self.link1.angularVelocity = initial_state[2]
            # print("just set angle to: {}, angVel to: {}".format(self.link1.angle, self.link1.angularVelocity))
            # print("this gives a current obs of: ", self.get_current_obs())
        if noisy:
            stds = np.array([0.1, 0.01])
            pos1, v1 = np.random.randn(*stds.shape) * stds
            self.link1.angle = pos1
            self.link1.angularVelocity = v1
        return self.get_current_obs()

    # def get_tip_pos(self):
    #     cur_center_pos = self.link1.position
    #     print("The link1 position should be 0: ", self.link1.position)
    #     cur_angle = self.link1.angle
    #     cur_pos = (
    #         cur_center_pos[0] - self.link_len*np.sin(cur_angle),
    #         cur_center_pos[1] - self.link_len*np.cos(cur_angle)
    #     )
    #     return cur_pos

    @overrides
    def compute_reward(self, action):
        yield
        tgt_pos = np.asarray([0, -1, 0])
        cur_pos = self.get_current_obs()
        if self.eps_sparse:
            rew = 1 if np.linalg.norm(cur_pos - tgt_pos) < self.eps_sparse else 0
        else:
            rew = -np.linalg.norm(cur_pos - tgt_pos)
        yield rew

    def is_current_done(self):
        return False
