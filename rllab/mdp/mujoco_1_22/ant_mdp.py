from mujoco_mdp import MujocoMDP
from rllab.core.serializable import Serializable
import numpy as np
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class AntMDP(MujocoMDP, Serializable):

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='Coefficient for the control cost')
    @autoargs.arg('impact_cost_coeff', type=float,
                  help='Coefficient for the impact cost')
    @autoargs.arg('survive_reward', type=float,
                  help='Reward for being alive')
    def __init__(self, ctrl_cost_coeff=0.0, impact_cost_coeff=0.0, survive_reward=0.0):
        path = self.model_path('ant.xml')
        super(AntMDP, self).__init__(path, frame_skip=1, ctrl_scaling=1)
        Serializable.__init__(
            self, ctrl_cost_coeff=ctrl_cost_coeff,
            impact_cost_coeff=impact_cost_coeff, survive_reward=survive_reward
        )
        init_qpos = np.zeros_like(self.model.data.qpos)
        # Taken from John's code
        init_qpos[0] = 0.0
        init_qpos[2] = 0.55
        init_qpos[8] = 1.0
        init_qpos[10] = -1.0
        init_qpos[12] = -1.0
        init_qpos[14] = 1.0
        self.init_qpos = init_qpos
        self._ctrl_cost_coeff = ctrl_cost_coeff
        self._impact_cost_coeff = impact_cost_coeff
        self._survive_reward = survive_reward

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flatten(),
            self.model.data.qvel.flatten(),
            np.clip(self.model.data.qfrc_constraint.flatten(), -10, 10),
            self.get_body_com("torso"),
        ]).reshape(-1)

    def step(self, state, action):
        self.set_state(state)
        com_before = self.get_body_com("torso")
        next_state = self.forward_dynamics(state, action, restore=False)
        com_after = self.get_body_com("torso")
        forward_reward = (com_after[0] - com_before[0]) / self.model.opt.timestep / self.frame_skip
        ctrl_cost = self._ctrl_cost_coeff * min(np.sum(np.square(action)), 10)
        impact_cost = self._impact_cost_coeff * min(np.sum(np.square(self.model.data.qfrc_constraint)), 10)
        survive_reward = self._survive_reward
        reward = forward_reward - ctrl_cost - impact_cost + survive_reward

        notdone = np.isfinite(next_state).all() and next_state[2] >= 0.2 and next_state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return next_state, ob, reward, done

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
