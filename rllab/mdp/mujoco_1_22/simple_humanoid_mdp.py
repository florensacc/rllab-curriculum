from .mujoco_mdp import MujocoMDP
import numpy as np
from rllab.core.serializable import Serializable
from rllab.sampler import parallel_sampler
from rllab.misc.overrides import overrides
from rllab.misc.ext import extract
from rllab.misc import logger
from rllab.misc import autoargs


class SimpleHumanoidMDP(MujocoMDP, Serializable):

    FILE = 'simple_humanoid.xml'

    @autoargs.arg('vel_deviation_cost_coeff', type=float,
                  help='cost coefficient for velocity deviation')
    @autoargs.arg('alive_bonus', type=float,
                  help='bonus reward for being alive')
    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for control inputs')
    @autoargs.arg('impact_cost_coeff', type=float,
                  help='cost coefficient for impact')
    @autoargs.arg('clip_impact_cost', type=float,
                  help='maximum value of impact cost')
    def __init__(
            self,
            vel_deviation_cost_coeff=1e-2,
            alive_bonus=0.2,
            ctrl_cost_coeff=1e-3,
            impact_cost_coeff=1e-5,
            clip_impact_cost=0.5,
            *args, **kwargs):
        self.vel_deviation_cost_coeff = vel_deviation_cost_coeff
        self.alive_bonus = alive_bonus
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.impact_cost_coeff = impact_cost_coeff
        self.clip_impact_cost = clip_impact_cost
        super(SimpleHumanoidMDP, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        data = self.model.data
        return np.concatenate([
            data.qpos.flat,
            data.qvel.flat,
            np.clip(data.cfrc_ext, -1, 1).flat,
            self.get_body_com("torso").flat,
        ])

    def _get_com(self):
        data = self.model.data
        mass = self.model.body_mass
        xpos = data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

    def step(self, state, action):
        next_state = self.forward_dynamics(state, action, restore=False)
        next_obs = self.get_current_obs()

        alive_bonus = self.alive_bonus
        data = self.model.data

        comvel = self.get_body_comvel("torso")

        lin_vel_reward = comvel[0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = .5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        impact_cost = min(
            .5 * self.impact_cost_coeff * np.sum(
                np.square(np.clip(data.cfrc_ext, -1, 1))),
            self.clip_impact_cost,
        )
        vel_deviation_cost = 0.5 * self.vel_deviation_cost_coeff * np.sum(
            np.square(comvel[1:]))
        reward = lin_vel_reward + alive_bonus - ctrl_cost - \
            impact_cost - vel_deviation_cost
        done = data.qpos[2] < 0.8 or data.qpos[2] > 2.0

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
