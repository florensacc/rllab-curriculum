from rllab.mdp.mujoco.mujoco_mdp import MujocoMDP
from rllab.core.serializable import Serializable
import numpy as np
from rllab.misc.overrides import overrides
from rllab.misc.ext import extract
from rllab.misc import logger
from rllab.sampler import parallel_sampler


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class HalfCheetahMDP(MujocoMDP, Serializable):

    FILE = 'half_cheetah.xml'

    def __init__(self, *args, **kwargs):
        super(HalfCheetahMDP, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flatten()[1:],
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        run_cost = -1 * self.get_body_comvel("torso")[0]
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False
        return next_obs, reward, done

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
