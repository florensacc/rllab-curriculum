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
            self.model.data.qvel.flat,
            # self.model.data.qfrc_passive.flatten(),
            self.get_body_com("torso").flat,
            self.get_body_comvel("torso").flat,
        ])

    @overrides
    def reset_mujoco(self):

        #self.model.data.qpos = np.array([
        #    0.01229506, -0.13276104, 0.05166619, 0.03299242, 0.06732391 -0.01524453,
        # -0.05708525, -0.13850914, -0.12913244])
        #self.model.data.qvel = np.array([0.00400523 -0.00190118 -0.00060655
        #  0.00823948  0.00568258  0.01036385  0.00283133 -0.00421921 -0.00620851
        self.model.data.qpos = np.random.randn(9) * 0.01
        self.model.data.qvel = np.random.randn(9) * 0.1
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl
        # forward for a while so that the cheetah lands
        # for _ in xrange(50):
        #     self.model.step()

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
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        passive_cost = 1e-5 * np.sum(np.square(self.model.data.qfrc_passive))
        run_cost = -1 * comvel[0]
        upright_cost = 1e-5 * smooth_abs(self.get_body_xmat("torso")[2, 2] - 1, 0.1)
        cost = ctrl_cost + passive_cost + run_cost + upright_cost
        reward = -cost
        reward = reward
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
        forward_progress = np.concatenate(parallel_sampler.run_map(worker_collect_stats))
        logger.record_tabular(
            'AverageForwardProgress', np.mean(forward_progress))
        logger.record_tabular(
            'MaxForwardProgress', np.max(forward_progress))
        logger.record_tabular(
            'MinForwardProgress', np.min(forward_progress))
        logger.record_tabular(
            'StdForwardProgress', np.std(forward_progress))
