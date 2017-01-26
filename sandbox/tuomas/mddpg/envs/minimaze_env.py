from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.spaces.box import Box

import numpy as np


class MultiRewardEnv(Env):
    """
    2D double integrator environment with two rewars: intrinsic velocity
    reward and external reward for reaching goal.
    state: position and velocity in 2D
    action: force
    """
    def __init__(self, goals=None, dynamics=None):
        self._max_force = 1
        self._max_speed = 0.5
        self._action_cost_coeff = 0.1
        self._speed_reward_coeff = 1
        self._hit_penalty = 1.
        # Goal reward is designed so that it is better for the exploration
        # critic to choose reward over avoiding it.
        self._goal_reward = 100.
        self._goal_tolerance = 0.25
        self._target_speed = 0.5

        if goals is None:
            goals = np.array(((0, 1),))
        self._goal_pos = goals

        if dynamics is None:
            dynamics = PointMassDynamics

        self._dynamics = dynamics(max_force=self._max_force,
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
        hit_speed = self._dynamics.step(action)
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
        reward_intrinsic = - (self._speed_reward_coeff *
                            np.abs(speed - self._target_speed) ** 2)
        reward_extrinsic = self._goal_reward if done else 0.0

        reward_intrinsic -= self._hit_penalty * hit_speed**2
        reward_extrinsic -= self._hit_penalty * hit_speed**2

        # Intrinsic penalty for violating action boundaries.
        action_penalty = max(0, action_norm_sq**0.5 - self._max_force)**2
        #print(str(np.array((action_penalty, hit_speed, reward_intrinsic, speed))))

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
        ax.set_xlim((-5, 5))
        ax.set_ylim((-1, 2))
        ax.grid(True)
        for goal in self._goal_pos:
            ax.plot(goal[0], goal[1], 'xk', mew=8, ms=16)

        self._dynamics.set_axis(ax)

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
        self._max_speed_so_far = 0.0

    def reset(self):
        self._state = self._init_state

    def observe(self):
        return self._state.copy()

    def _clip_force(self, action):
        action_norm = np.linalg.norm(action)
        if action_norm > self._max_force:
            action = action / action_norm * self._max_force

        return action

    def _clip_speed(self, velocity):
        speed = np.linalg.norm(velocity)
        if speed > self._max_speed_so_far:
            self._max_speed_so_far = speed
            #print('Speed: ' + str(speed))
        if speed > self._max_speed:
            velocity = velocity / speed * self._max_speed
        return velocity

    def step(self, action):
        action = self._clip_force(action)
        self._state = self._A.dot(self._state) + self._B.dot(action)

        self._state[2:] = self._clip_speed(self._state[2:])


    def set_axis(self, ax):
        pass

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


class Minimaze(PointMassDynamics):
    def __init__(self, *args, **kwargs):
        super(Minimaze, self).__init__(*args, **kwargs)

        self._eps = 1e-3

        # Horizontal walls (y-coord, x_start, x_end)
        self._h_walls = np.array([[1.5, -4, 4],
                                  [0.5, -3, 3],
                                  [-0.5, -4, 4]])
        # Vertical walls (x-coord, y_start, y_end)
        self._v_walls = np.array([[-4, -0.5, 1.5],
                                  [-0.5, -0.5, 0.5],
                                  [4, -0.5, 1.5]])

    def step(self, action):
        action = self._clip_force(action)
        next_state = self._A.dot(self._state) + self._B.dot(action)

        # Check if next state violates any constraints (walls)
        xp, yp = self._state[:2]
        xn, yn, dxn, dyn = next_state

        hit_speed = 0.
        for y, x_start, x_end in self._h_walls:
            if x_start < xn < x_end or x_start < xp < x_end:
                if yp > y > yn:
                    yn = y + self._eps
                    hit_speed = abs(dyn)
                    dyn = 0
                elif yp < y < yn:
                    yn = y - self._eps
                    hit_speed = abs(dyn)
                    dyn = 0

        for x, y_start, y_end in self._v_walls:
            if y_start < yn < y_end or y_start < yp < y_end:
                if xp < x < xn:
                    xn = x - self._eps
                    hit_speed = abs(dxn)
                    dxn = 0
                elif xp > x > xn:
                    xn = x + self._eps
                    hit_speed = abs(dxn)
                    dxn = 0

        self._state = np.array((xn, yn, dxn, dyn))
        self._state[2:] = self._clip_speed(self._state[2:])
        return hit_speed

    @overrides
    def set_axis(self, ax):
        for y, x_start, x_end in self._h_walls:
            ax.plot((x_start, x_end), (y, y), 'k', linewidth=2)
        for x, y_start, y_end in self._v_walls:
            ax.plot((x, x), (y_start, y_end), 'k', linewidth=2)







