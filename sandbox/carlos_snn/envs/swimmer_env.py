from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs


class SwimmerEnv(MujocoEnv, Serializable):
    FILE = 'swimmer.xml'

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            ctrl_cost_coeff=1e-2,
            *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        super(SwimmerEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).reshape(-1)

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        forward_reward = np.linalg.norm(self.get_body_comvel("torso"))  # swimmer has no problem of jumping reward
        reward = forward_reward - ctrl_cost
        done = False
        return Step(next_obs, reward, done)

    @overrides
    def log_diagnostics(self, paths):
        # instead of just path["obs"][-1][-3] we will look at the distance to origin
        progs = [
            np.linalg.norm(path["observations"][-1][-3:-1] - path["observations"][0][-3:-1])  # gives (x,y) coord -not last z
            for path in paths
            ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))
        # now we will grid the space and check how much of it the policy is covering
        visitation = np.zeros((120, 120))  # we assume the furthest it can go is 6 Check it!!
        for path in paths:
            com_x = np.clip(((np.array(path['observations'][:][-3]) + 6) * 10).astype(int), 0, 119)
            com_y = np.clip(((np.array(path['observations'][:][-2]) + 6) * 10).astype(int), 0, 119)
            coms = zip(com_x, com_y)
            for com in coms:
                visitation[com] += 1
        print visitation
        total_visitation = np.count_nonzero(visitation)
        logger.record_tabular('VisitationTotal', total_visitation)
