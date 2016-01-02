from rllab.mdp.box2d.box2d_mdp import Box2DMDP
from rllab.mdp.box2d.parser import find_body, find_joint
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class HalfCheetahMDP(Box2DMDP, Serializable):

    @autoargs.inherit(Box2DMDP.__init__)
    def __init__(self, **kwargs):
        kwargs["frame_skip"] = kwargs.get("frame_skip", 10)
        super(HalfCheetahMDP, self).__init__(self.model_path("half_cheetah.xml"))
        Serializable.__init__(self, **kwargs)
        self.torso = find_body(self.world, "torso")

    @overrides
    def forward_dynamics(self, state, action, restore=True):
        lb, ub = super(HalfCheetahMDP, self).action_bounds
        forces = []
        for i, ctrl, act in zip(
                xrange(len(action)),
                self.extra_data.controls,
                action
        ):
            joint = find_joint(self.world, ctrl.joint)
            spring = 2./np.pi * np.arctan(-2.*joint.angle-0.05*joint.speed)
            limit = min([1, max([-1, spring + action[i]])])
            forces.append(limit * ub[i] * 1.)
        return super(HalfCheetahMDP, self).forward_dynamics(state, np.array(forces), restore)

    @property
    @overrides
    def action_bounds(self):
        b = np.ones(self.action_dim)
        return b*-2, b*2

    @overrides
    def get_current_obs(self):
        raw_obs = self.get_raw_obs()
        return raw_obs

    @overrides
    def get_current_reward(
            self, state, raw_obs, action, next_state, next_raw_obs):
        return 0

    @overrides
    def is_current_done(self):
        return False
