from rllab.mdp.mujoco_1_22.mujoco_mdp import MujocoMDP
from rllab.core.serializable import Serializable
import numpy as np
from rllab.misc.overrides import overrides
from rllab.sampler import parallel_sampler
from rllab.misc.ext import extract
from rllab.misc import logger


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
            self.model.data.qfrc_passive.flat,
            self.get_body_com("torso").flat,
        ])

    def step(self, state, action):
        self.set_state(state)
        next_state = self.forward_dynamics(state, action, restore=False)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 1e-1 * np.sum(np.square(action / scaling))
        passive_cost = 1e-5 * np.sum(np.square(self.model.data.qfrc_passive))
        forward_reward = self.dcom[0] / self.timestep / self.frame_skip
        upright_cost = 1e-5 * smooth_abs(
            self.get_body_xmat("torso")[2, 2] - 1, 0.1)
        reward = forward_reward - ctrl_cost - passive_cost - upright_cost
        qpos = self.model.data.qpos
        done = not (qpos[0] > 0.8 and qpos[0] < 2.0
                    and qpos[2] > -1.0 and qpos[2] < 1.0)
        return next_state, next_obs, reward, done

    @overrides
    def log_extra(self):
        stats = parallel_sampler.run_map(_worker_collect_stats)
        mean_progs, max_progs, min_progs, std_progs = extract(
            stats,
            "mean_prog", "max_prog", "min_prog", "std_prog"
        )
        logger.record_tabular('AverageForwardProgress', np.mean(mean_progs))
        logger.record_tabular('MaxForwardProgress', np.max(max_progs))
        logger.record_tabular('MinForwardProgress', np.min(min_progs))
        logger.record_tabular('StdForwardProgress', np.mean(std_progs))


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
