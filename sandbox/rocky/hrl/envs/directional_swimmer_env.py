


from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs


DIRECTIONS = np.array([0., np.pi / 2, np.pi, np.pi * 3 / 2])


class DirectionalSwimmerEnv(MujocoEnv, Serializable):
    FILE = 'swimmer.xml'

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            ctrl_cost_coeff=1e-2,
            turn_interval=100,
            *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        super(DirectionalSwimmerEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        self.direction_t = None
        self.direction = None
        self.forward_progress = None
        self.turn_interval = turn_interval
        self.reset()

    def reset(self):
        self.direction_t = 0
        self.direction = np.random.choice(DIRECTIONS)
        self.forward_progress = 0.
        MujocoEnv.reset(self)
        return self.get_current_obs()

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
            [np.sin(self.direction), np.cos(self.direction)],
        ]).reshape(-1)

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        x_vel, y_vel = self.get_body_comvel("torso")[:2]
        forward_reward = x_vel * np.sin(self.direction) + y_vel * np.cos(self.direction)
        self.forward_progress += forward_reward
        reward = forward_reward - ctrl_cost
        done = False
        self.direction_t += 1
        if self.direction_t % self.turn_interval == 0:
            self.direction_t = 0
            self.direction = np.random.choice(DIRECTIONS)
            # self.direction = np.random.uniform(0., 2 * np.pi)
        return Step(next_obs, reward, done, forward_progress=self.forward_progress)

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["env_infos"]["forward_progress"][-1]
            for path in paths
            ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))
