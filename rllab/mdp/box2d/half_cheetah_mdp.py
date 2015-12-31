from rllab.mdp.box2d.box2d_mdp import Box2DMDP
from rllab.mdp.box2d.parser import find_body
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides


class HalfCheetahMDP(Box2DMDP, Serializable):

    def __init__(self):
        super(HalfCheetahMDP, self).__init__(self.model_path("half_cheetah.xml"))
        Serializable.__init__(self)

    @overrides
    def get_current_obs(self):
        raw_obs = self.get_raw_obs()
        # remove the x position from the observation
        return raw_obs

    @overrides
    def get_current_reward(
            self, state, raw_obs, action, next_state, next_raw_obs):
        return 0

    @overrides
    def is_current_done(self):
        return False
