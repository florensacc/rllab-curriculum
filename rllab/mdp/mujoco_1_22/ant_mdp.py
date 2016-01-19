from mujoco_mdp import MujocoMDP
from rllab.core.serializable import Serializable
import numpy as np
from rllab.misc.overrides import overrides


class AntMDP(MujocoMDP, Serializable):

    def __init__(self):
        path = self.model_path('ant.xml')
        super(AntMDP, self).__init__(path, frame_skip=1, ctrl_scaling=1)
        init_qpos = np.zeros_like(self.model.data.qpos)
        # Taken from John's code
        init_qpos[0] = 0.0
        init_qpos[2] = 0.55
        init_qpos[8] = 1.0
        init_qpos[10] = -1.0
        init_qpos[12] = -1.0
        init_qpos[14] = 1.0
        self.init_qpos = init_qpos
        self._ctrl_costs = []
        self._impact_costs = []
        self._forward_rewards = []

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
        ctrl_cost = 0#1e-6 * np.sum(np.square(action))
        impact_cost = 0#1e-3 * 0.5 * np.sum(np.square(np.clip(self.model.data.qfrc_constraint, -10, 10)))
        survive_reward = 0#0.05
        reward = forward_reward - ctrl_cost - impact_cost + survive_reward

        self._ctrl_costs.append(ctrl_cost)
        self._impact_costs.append(impact_cost)
        self._forward_rewards.append(forward_reward)
        if len(self._ctrl_costs) > 10000:
            print "Avg ctrl cost: %f" % np.mean(np.abs(self._ctrl_costs))
            print "Avg impact cost: %f" % np.mean(np.abs(self._impact_costs))
            print "Avg forward cost: %f" % np.mean(np.abs(self._forward_rewards))
            self._ctrl_costs = []
            self._impact_costs = []
            self._forward_rewards = []

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
