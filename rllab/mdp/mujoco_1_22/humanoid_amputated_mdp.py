from .mujoco_mdp import MujocoMDP
import numpy as np
from rllab.core.serializable import Serializable
from rllab.sampler import parallel_sampler
from rllab.misc.overrides import overrides
from rllab.misc.ext import extract
from rllab.misc import logger


class HumanoidAmputatedMDP(MujocoMDP, Serializable):

    FILE = 'humanoid_amputated.xml'

    def __init__(self, *args, **kwargs):
        super(HumanoidAmputatedMDP, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        data = self.model.data
        return np.concatenate([
            data.qpos.flat,
            data.qvel.flat,
            # data.cinert.flat,
            # data.cvel.flat,
            # data.qfrc_actuator.flat,
            data.cfrc_ext.flat,
            self.get_body_com("torso").flat,
        ])

    def _get_com(self):
        data = self.model.data
        mass = self.model.body_mass
        xpos = data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

    def step(self, state, action):
        # self.set_state(state)
        # before_center = self._get_com()
        # self.model.forward()
        # before_com = self.get_body_com("front")
        next_state = self.forward_dynamics(state, action, restore=False)
        next_obs = self.get_current_obs()
        # after_center = self._get_com()

        alive_bonus = 1.0
        data = self.model.data
        # mass = self.model.body_mass
        # xpos = data.xipos
        # after_center = (np.sum(mass * xpos, 0) / np.sum(mass))[0]
        lin_vel_reward = 1 * self.get_body_comvel("torso")[0]
        quad_ctrl_cost = .5 * 1e-5 * np.sum(np.square(data.ctrl))
        quad_impact_cost = .5 * 1e-5 * np.sum(np.square(data.cfrc_ext))
        quad_impact_cost = min(10.0, quad_impact_cost)
        reward = lin_vel_reward + alive_bonus - quad_ctrl_cost - \
            quad_impact_cost
        done = data.qpos[2] < 0.9 or data.qpos[2] > 2.0

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
