from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs


class AntEnv(MujocoEnv, Serializable):
    FILE = 'ant.xml'

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(self,
                 ctrl_cost_coeff=1e-2,
                 *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        super(AntEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)  # locals()????

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

    def step(self, action):
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        forward_reward = np.linalg.norm(comvel[0:-1])  # instead of comvel[0] (does this give jumping reward??)
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),  # what is this??
        survive_reward = 0.05  # this is not in swimmer neither!!
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            np.linalg.norm(path["observations"][-1][-3:-1] - path["observations"][0][-3:-1])
            for path in paths
            ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))
        # now we will grid the space and check how much of it the policy is covering
        furthest = 100
        c_grid = furthest * 10 * 2
        visitation = np.zeros((c_grid, c_grid))  # we assume the furthest it can go is 60 Check it!!
        for path in paths:
            com_x = np.clip(((np.array(path['observations'][:][-3]) + furthest) * 10).astype(int), 0, c_grid -1)
            com_y = np.clip(((np.array(path['observations'][:][-2]) + furthest) * 10).astype(int), 0, c_grid -1)
            coms = zip(com_x, com_y)
            for com in coms:
                visitation[com] += 1
        total_visitation = np.count_nonzero(visitation)
        logger.record_tabular('VisitationTotal', total_visitation)
