from rllab.mdp.box2d.box2d_mdp import Box2DMDP
from rllab.mdp.box2d.parser import find_body
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides


class HopperMDP(Box2DMDP, Serializable):

    def __init__(self, *args, **kwargs):
        super(HopperMDP, self).__init__(
            self.model_path("hopper.xml"), *args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        self.torso = find_body(self.world, "torso")
        self.thigh = find_body(self.world, "thigh")
        self.leg = find_body(self.world, "leg")
        self.foot = find_body(self.world, "foot")

    @overrides
    def compute_reward(self, action):
        before_com = self.get_com_position("torso", "thigh", "leg", "foot")
        yield
        after_com = self.get_com_position("torso", "thigh", "leg", "foot")
        if self.is_current_done():
            yield 0
        else:
            yield (after_com[0] - before_com[0]) / self.timestep

    @overrides
    def is_current_done(self):
        raw_obs = super(HopperMDP, self).get_raw_obs()
        # TODO add the condition for forward pitch
        notdone = np.isfinite(raw_obs).all() and \
            (np.abs(raw_obs) < 100).all() and \
            (self.torso.position[1] > 0.7)
        return not notdone
