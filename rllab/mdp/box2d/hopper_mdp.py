from rllab.mdp.box2d.box2d_mdp import Box2DMDP
from rllab.mdp.box2d.parser import find_body
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides


class HopperMDP(Box2DMDP, Serializable):

    def __init__(self):
        super(HopperMDP, self).__init__(self.model_path("hopper.xml"))
        Serializable.__init__(self)
        self.torso = find_body(self.world, "torso")
        self.thigh = find_body(self.world, "thigh")
        self.leg = find_body(self.world, "leg")
        self.foot = find_body(self.world, "foot")

    @overrides
    def get_current_obs(self):
        raw_obs = self.get_raw_obs()
        # remove the x position from the observation
        return raw_obs[1:]

    @overrides
    def get_current_reward(
            self, state, raw_obs, action, next_state, next_raw_obs):
        if self.is_current_done():
            return 0
        return (next_raw_obs[0] - raw_obs[0]) / self.timestep

    @overrides
    def is_current_done(self):
        raw_obs = self.get_raw_obs()
        # TODO add the condition for forward pitch
        notdone = np.isfinite(raw_obs).all() and \
            (np.abs(raw_obs[1:]) < 100).all() and \
            (self.torso.position[1] > 0.7)
        return not notdone
