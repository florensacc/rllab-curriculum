from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
import numpy as np

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger


class AntEnv(MujocoEnv, Serializable):
    FILE = 'ant_target.xml'

    def __init__(self, append_goal=False, *args, **kwargs):
        self.append_goal = append_goal
        super(AntEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        pos = self.model.data.qpos.flat[:-2]
        vel = self.model.data.qvel.flat[:-2]
        current_goal = self.model.data.qpos.flat[-2:].reshape(-1)
        non_goal_obs = np.concatenate([
            pos, vel,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)
        if self.append_goal:
            return np.concatenate([non_goal_obs, current_goal])
        else:
            return non_goal_obs

    def step(self, action):
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        forward_reward = comvel[0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() and 1.0 >= float(state[2]) >= 0.3
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)

    @overrides
    def reset(self, goal=(2, 0), init_state=None, *args, **kwargs):  # can't use mujoco.reset because qacc error...
        # super(AntEnv, self).reset(*args, **kwargs)
        # self.model.data.qpos = np.concatenate([self.model.data.qpos[:-2, :], np.array(goal).reshape((2, 1)).astype(float)])
        # self.model.data.qvel = np.concatenate([self.model.data.qvel[:-2, :], np.zeros((2, 1)).astype(float)])
        if init_state is None:
            if self.random_init_state:
                self.model.data.qpos = self.init_qpos + \
                                       np.random.normal(size=self.init_qpos.shape) * 0.01
                self.model.data.qvel = self.init_qvel + \
                                       np.random.normal(size=self.init_qvel.shape) * 0.1
            else:
                self.model.data.qpos = self.init_qpos
                self.model.data.qvel = self.init_qvel

            self.model.data.qacc = self.init_qacc
            self.model.data.ctrl = self.init_ctrl
        else:
            start = 0
            for datum_name in ["qpos", "qvel", "qacc", "ctrl"]:
                datum = getattr(self.model.data, datum_name)
                datum_dim = datum.shape[0]
                datum = init_state[start: start + datum_dim]
                setattr(self.model.data, datum_name, datum)
                start += datum_dim

        self.model.data.qpos = np.concatenate([np.array(self.model.data.qpos[:-2, :], dtype=float),
                                               np.array(goal).reshape((2, 1)).astype(float)])
        self.model.data.qvel = np.concatenate([np.array(self.model.data.qvel[:-2, :], dtype=float),
                                               np.zeros((2, 1)).astype(float)])

        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))
