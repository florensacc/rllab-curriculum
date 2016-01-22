from rllab.mdp.mujoco_1_22.mujoco_mdp import MujocoMDP
from rllab.core.serializable import Serializable
import numpy as np
from rllab.misc.overrides import overrides


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class Walker2DMDP(MujocoMDP, Serializable):

    FILE = 'walker2d.xml'

    def __init__(self, *args, **kwargs):
        super(Walker2DMDP, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            # self.model.data.qfrc_passive.flat,
            self.get_body_com("torso").flat,
        ])

    def step(self, state, action):
        self.set_state(state)
        prev_com = self.get_body_com("torso")
        next_state = self.forward_dynamics(state, action, restore=False)
        after_com = self.get_body_com("torso")

        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        ctrl_cost = 0  # 1e-1 * np.sum(np.square(action))
        passive_cost = 0  # 1e-5 * np.sum(np.square(self.model.data.qfrc_passive))

        run_cost = -1 * (after_com[0] - prev_com[0]) / self.model.opt.timestep
        upright_cost = 0#1e-5 * smooth_abs(self.get_body_xmat("torso")[2, 2] - 1, 0.1)
        cost = ctrl_cost + passive_cost + run_cost + upright_cost
        reward = -cost
        qpos = self.model.data.qpos
        done = not (qpos[0] > 0.8 and qpos[0] < 2.0 and qpos[2] > -1.0 and qpos[2] < 1.0)
        return next_state, next_obs, reward, done

    @overrides
    def log_extra(self, logger, paths):
        forward_progress = \
            [path["observations"][-1][-3] - path["observations"][0][-3] for path in paths]
        logger.record_tabular(
            'AverageForwardProgress', np.mean(forward_progress))
        logger.record_tabular(
            'MaxForwardProgress', np.max(forward_progress))
        logger.record_tabular(
            'MinForwardProgress', np.min(forward_progress))
        logger.record_tabular(
            'StdForwardProgress', np.std(forward_progress))
