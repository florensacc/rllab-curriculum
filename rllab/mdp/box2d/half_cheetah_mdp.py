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
        self.ctrl_joints = [find_joint(self.world, ctrl.joint) for ctrl in self.extra_data.controls]


    def get_spring_forces(self, state):
        with self._set_state_tmp(state):
            springs = [
                2./np.pi * np.arctan(-2.*joint.angle-0.05*joint.speed) * 1
                for joint in self.ctrl_joints
            ]
            return np.array(springs)

    def get_torques(self, x):
        lb, ub = super(HalfCheetahMDP, self).action_bounds
        return x * ub * 0.01


    @overrides
    def forward_dynamics(self, state, action, restore=True):
        clipped = np.clip(action+self.get_spring_forces(state), -1, 1)
        forces = self.get_torques(clipped)
        # forces = []
        # for i, ctrl, act in zip(
        #         xrange(len(action)),
        #         self.extra_data.controls,
        #         action
        # ):
        #     joint = find_joint(self.world, ctrl.joint)
        #     spring = 2./np.pi * np.arctan(-2.*joint.angle-0.05*joint.speed) * 1
        #     limit = min([1, max([-1, spring + action[i]])])
        #     # print limit
        #     forces.append(limit * ub[i] * 0.01)
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
            springs = self.get_spring_forces(state)
            outofrange_penalty = -0.05*np.sum(np.clip(np.abs(springs+action)-1, -10, 0))
            torque_penalty = -0.5*np.sum(np.clip(
                np.abs(self.get_torques(np.clip(springs+action, -1, 1))),
                0,
                0.05
            ))

            return 0.5*speed + touch_ground_penalty + alive_bonus + outofrange_penalty + torque_penalty

    @overrides
    def is_current_done(self):
        if self.torso.position[1] <= 0.3 or self.fthigh.position[1] <= 0.3:
            return True
        return False
