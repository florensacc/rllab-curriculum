from rllab.mdp.box2d.box2d_mdp import Box2DMDP
from rllab.mdp.box2d.parser import find_body, find_joint
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class HalfCheetahMDP(Box2DMDP, Serializable):

    timestep = 0.001

    @autoargs.inherit(Box2DMDP.__init__)
    def __init__(self, **kwargs):
        kwargs["frame_skip"] = kwargs.get("frame_skip", 1)
        super(HalfCheetahMDP, self).__init__(self.model_path("half_cheetah.xml"), **kwargs)
        Serializable.__init__(self, **kwargs)
        self.torso = find_body(self.world, "torso")
        self.bshin = find_body(self.world, "bshin")
        self.bfoot = find_body(self.world, "bfoot")
        self.fthigh = find_body(self.world, "fthigh")

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
            spring = 0#2./np.pi * np.arctan(-2.*joint.angle-0.05*joint.speed) * 0.25
            limit = min([1, max([-1, spring + action[i]])])
            # print limit
            forces.append(limit * ub[i] * 0.01)
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
        def soft_step(x):
            if x <= 0:
                return 0
            elif x <= 0.5:
                return 2*x**2
            elif x <= 1:
                return 1-2*(x-1)**2
            else:
                return 1
        with self._set_state_tmp(state):
            alive_bonus = 0.1
            speed = self.torso.linearVelocity[0]
            touch_ground_penalty = (soft_step(2*self.torso.position[1])-1) + \
                                   (soft_step(9*self.bfoot.position[1])-1) + \
                                   (soft_step(4*self.bshin.position[1])-1)
            return 0.5*speed + touch_ground_penalty + alive_bonus

    @overrides
    def is_current_done(self):
        if self.torso.position[1] <= 0.3 or self.fthigh.position[1] <= 0.3:
            return True
        return False
