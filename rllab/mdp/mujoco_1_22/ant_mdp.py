from mujoco_mdp import MujocoMDP
from rllab.core.serializable import Serializable
import numpy as np
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc.ext import extract
from rllab.sampler import parallel_sampler


class AntMDP(MujocoMDP, Serializable):

    FILE = 'ant.xml'

    def __init__(self, *args, **kwargs):
        super(AntMDP, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        init_qpos = np.zeros_like(self.model.data.qpos)
        # Taken from John's code
        init_qpos[0] = 0.0
        init_qpos[2] = 0.55
        init_qpos[8] = 1.0
        init_qpos[10] = -1.0
        init_qpos[12] = -1.0
        init_qpos[14] = 1.0
        self.init_qpos = init_qpos

    # def get_current_obs(self):
    #     return np.concatenate([
    #         self.model.data.qpos.flatten(),
    #         self.model.data.qvel.flatten(),
    #         self.model.data.qfrc_constraint.flatten(),
    #         self.get_body_com("torso"),
    #     ]).reshape(-1)

    def step(self, state, action):
        next_state = self.forward_dynamics(state, action, restore=False)
        forward_reward = (self.dcom[0]) \
            / self.model.opt.timestep / self.frame_skip
        ctrl_cost = 0.5 * 1e-5 * np.sum(np.square(action))
        impact_cost = min(
            0.5 * 1e-5 * np.sum(np.square(self.model.data.qfrc_constraint)),
            10.0
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - impact_cost + survive_reward
        notdone = np.isfinite(next_state).all() \
            and next_state[2] >= 0.2 and next_state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return next_state, ob, reward, done

    @staticmethod
    def _worker_collect_stats():
        PG = parallel_sampler.G
        paths = PG.paths
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        return dict(
            mean_prog=np.mean(progs),
            max_prog=np.max(progs),
            min_prog=np.min(progs),
            std_prog=np.std(progs),
        )

    @overrides
    def log_extra(self):
        stats = parallel_sampler.run_map(AntMDP._worker_collect_stats)
        mean_progs, max_progs, min_progs, std_progs = extract(
            stats,
            "mean_prog", "max_prog", "min_prog", "std_prog"
        )
        logger.record_tabular('AverageForwardProgress', np.mean(mean_progs))
        logger.record_tabular('MaxForwardProgress', np.max(max_progs))
        logger.record_tabular('MinForwardProgress', np.min(min_progs))
        logger.record_tabular('StdForwardProgress', np.mean(std_progs))
