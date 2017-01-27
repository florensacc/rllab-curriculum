from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.spaces.box import Box

import numpy as np
import matplotlib.pyplot as plt


class MultiRewardEnv(Env):
    """
    2D double integrator environment with two rewars: intrinsic velocity
    reward and external reward for reaching goal.
    state: position and velocity in 2D
    action: force
    """
    def __init__(self, goals=None):
        self._max_force = 1
        self._max_speed = 0.5
        self._action_cost_coeff = 1
        self._speed_reward_coeff = 0.01
        # Goal reward is designed so that it is better for the exploration
        # critic to choose reward over avoiding it.
        self._goal_reward = 50.
        self._goal_tolerance = 1.

        if goals is None:
            goals = np.array(((5, 5),
                              (5, -5),
                              (-5, 5),
                              (-5, -5)))
        self._goal_pos = goals

        self._dynamics = PointMassDynamics(max_force=self._max_force,
                                           max_speed=self._max_speed)

    def reset(self):
        self._dynamics.reset()
        return self._dynamics.observe()

    @property
    def observation_space(self):
        return Box(
            low=np.inf,
            high=np.inf,
            shape=(4,)
        )

    @property
    def action_space(self):
        return Box(
            low=-self._max_force,
            high=self._max_force,
            shape=(2,)
        )

    @property
    def reward_dim(self):
        return 2

    # TODO: is this used anywhere?
    def get_current_obs(self):
        return self._dynamics.observe()

    def step(self, action):
        action = action.flatten()

        # Advance.
        self._dynamics.step(action)
        next_obs = self._dynamics.observe()

        # Check if reached the goal.
        d = []
        for goal in self._goal_pos:
            d.append(np.linalg.norm(self._dynamics.position - goal))
        dist_to_goal = min(d)
        #dist_to_goal = np.linalg.norm(self._dynamics.position - self._goal_pos)
        done = dist_to_goal < self._goal_tolerance

        # Compute rewards / costs.
        action_norm_sq = np.sum(action**2)
        action_cost = self._action_cost_coeff * action_norm_sq
        speed = np.linalg.norm(self._dynamics.velocity)
        reward_intrinsic = self._speed_reward_coeff * speed
        reward_extrinsic = self._goal_reward if done else 0.0

        # Intrinsic penalty for violating action boundaries.
        action_penalty = max(0, action_norm_sq**0.5 - self._max_force)**2

        rewards = np.array(
            (reward_intrinsic - action_penalty,
             reward_extrinsic - action_cost)
        )

        return Step(next_obs, rewards, done, pos=next_obs[:2])

    @staticmethod
    def plot_path(env_info_list, ax, style='b'):
        path = np.concatenate([i['pos'][None] for i in env_info_list], axis=0)
        xx = path[:, 0]
        yy = path[:, 1]
        line, = ax.plot(xx, yy, style)
        return line

    def set_axis(self, ax):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim((-10, 10))
        ax.set_ylim((-10, 10))
        ax.grid(True)
        for goal in self._goal_pos:
            ax.plot(goal[0], goal[1], 'xk', mew=8, ms=16)

    @overrides
    def log_diagnostics(self, paths):
        pass

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass

    def log_stats(self, epoch, paths):
        raise NotImplementedError

        n_goal = len(self.goal_positions)
        goal_reached = [False] * n_goal

        for path in paths:
            last_obs = path["observations"][-1]
            for i, goal in enumerate(self.goal_positions):
                if np.linalg.norm(last_obs - goal) < self.goal_threshold:
                    goal_reached[i] = True

        stats = {
            "env:goal_reached": goal_reached.count(True)
        }
        return stats

    def __getstate__(self):
       return dict(dynamics=self._dynamics.__getstate__())

    def __setstate__(self, d):
        self._dynamics.__setstate__(d['dynamics'])


class PointMassDynamics(object):

    def __init__(self, max_force, max_speed, mass=10):
        self._max_force = max_force
        self._max_speed = max_speed
        self._A = np.array([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self._B = np.array([[0, 0],
                            [0, 0],
                            [1./mass, 0],
                            [0, 1./mass]])

        self._init_state = np.zeros((4,))
        self._state = self._init_state

    def reset(self):
        self._state = self._init_state

    def observe(self):
        return self._state.copy()

    def _clip_force(self, action):
        action_norm = np.linalg.norm(action)
        if action_norm > self._max_force:
            action = action / action_norm * self._max_force

        return action

    def step(self, action):
        action = self._clip_force(action)
        self._state = self._A.dot(self._state) + self._B.dot(action)

        speed = np.linalg.norm(self._state[2:])
        if speed > self._max_speed:
            self._state[2:] = self._state[2:] / speed * self._max_speed

    @property
    def velocity(self):
        return self._state[2:]

    @property
    def position(self):
        return self._state[:2]

    def __getstate__(self):
        d = dict(state=self._state)
        return d

    def __setstate__(self, d):
        self._state = d['state']


