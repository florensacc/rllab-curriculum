from rllab.mdp.mujoco_1_22.mujoco_mdp import MujocoMDP
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
        self._initial_com = self.get_body_com("torso")

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flatten()[1:],
            self.model.data.qvel.flatten(),
            # self.model.data.qfrc_passive.flatten(),
            #self.get_body_com("torso").flatten(),
        ])

    @overrides
    def reset_mujoco(self):
        self.model.data.qpos = np.random.randn(9) * 0.01
        self.model.data.qvel = np.random.randn(9) * 0.1
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, state, action):
        next_state = self.forward_dynamics(state, action, restore=False)
        comvel = self.get_body_comvel("torso")

        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        ctrl_cost = 1e-1 * np.sum(np.square(action))
        passive_cost = 0  # 1e-5 * np.sum(np.square(self.model.data.qfrc_passive))
        run_cost = -1 * comvel[0]
        upright_cost = 0  # 1e-5 * smooth_abs(self.get_body_xmat("torso")[2, 2] - 1, 0.1)
        cost = ctrl_cost + passive_cost + run_cost + upright_cost
        reward = -cost
        done = False#self.model.data.qpos[1] < -0.2
        #done = False  # after_com[0] < self._initial_com[0] - 0.1 # False
        return next_state, next_obs, reward, done

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
        return
        stats = parallel_sampler.run_map(HalfCheetahMDP._worker_collect_stats)
        mean_progs, max_progs, min_progs, std_progs = extract(
            stats,
            "mean_prog", "max_prog", "min_prog", "std_prog"
        )
        logger.record_tabular('AverageForwardProgress', np.mean(mean_progs))
        logger.record_tabular('MaxForwardProgress', np.max(max_progs))
        logger.record_tabular('MinForwardProgress', np.min(min_progs))
        logger.record_tabular('StdForwardProgress', np.mean(std_progs))
