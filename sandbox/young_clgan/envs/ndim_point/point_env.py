from rllab.envs.base import Step, Env
from rllab.spaces.box import Box
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import logger
import numpy as np
import math
import random

UB = 2


class PointEnv(Env, Serializable):

    def __init__(self, dim=2, state_bounds=None, action_bounds=None,
                 control_mode='linear', *args, **kwargs):
        """
        :param dim: dimension of the position space. the obs will be 2*dim
        :param state_bounds: list or np.array of 2*dim with the ub of the state space
        :param action_bounds: list or np.array of 2*dim with the ub of the action space
        :param goal_generator: Proceedure to sample the and keep the goals
        :param reward_dist_threshold:
        :param control_mode:
        """
        Serializable.quick_init(self, locals())
        self.dim = dim
        self.control_mode = control_mode
        self.dt = 0.02
        self.pos = np.zeros(dim)
        self.vel = np.zeros(dim)
        self.state_ub = UB * np.ones(self.dim * 2) if state_bounds is None else np.array(state_bounds)
        self.action_ub = UB * np.ones(self.dim) if action_bounds is None else np.array(action_bounds)
        self._observation_space = Box(-1 * self.state_ub, self.state_ub)
        self._action_space = Box(-1 * self.action_ub, self.action_ub)

    @overrides
    def step(self, action):
        # print("action: ", action)
        if self.control_mode == 'linear':  # action is directly the acceleration
            dv = action * self.dt
            self.vel = np.clip(self.vel + dv, -self.state_ub[-self.dim:], self.state_ub[-self.dim:])
            self.pos = np.clip(self.pos + self.vel * self.dt, -self.state_ub[:self.dim], self.state_ub[:self.dim])
        else:
            raise NotImplementedError("Control mode not supported!")

        reward_ctrl = - np.square(action).sum()
        reward = reward_ctrl

        ob = self.get_current_obs()
        # print("observation: ", ob)
        return Step(
            ob, reward, done=False,
            reward_ctrl=reward_ctrl,
        )

    @overrides
    def reset(self, pos=None, vel=None, **kwargs):
        if pos is None:
            pos = np.zeros(self.dim)
        if vel is None:
            vel = np.zeros(self.dim)
        self.set_state(pos, vel)
        return self.get_current_obs()

    @overrides
    @property
    def action_space(self):
        return self._action_space

    @overrides
    @property
    def observation_space(self):
        return self._observation_space

    def is_feasible(self, goal):  # this is only for the position!!
        return self.observation_space.contains(np.concatenate((goal, np.zeros_like(goal))))

    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.pos,
            self.vel
        ])

    def set_state(self, pos, vel):
        self.pos = np.clip(pos, -self.state_ub[:self.dim], self.state_ub[:self.dim])
        self.vel = np.clip(vel, -self.state_ub[-self.dim:], self.state_ub[-self.dim:])

