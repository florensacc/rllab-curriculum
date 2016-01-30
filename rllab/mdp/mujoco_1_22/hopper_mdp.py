from rllab.mdp.mujoco_1_22.mujoco_mdp import MujocoMDP
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.sampler import parallel_sampler


PG = parallel_sampler.G

# states: [
# 0: z-coord,
# 1: x-coord (forward distance),
# 2: forward pitch along y-axis,
# 6: z-vel (up = +),
# 7: xvel (forward = +)

def worker_collect_stats():
    paths = PG.paths
    forward_progress = \
        np.array([path["states"][-1][1] - path["states"][0][1] for path in paths])
    return forward_progress#np.mean(np.exp(log_stds))

class HopperMDP(MujocoMDP, Serializable):

    FILE = 'hopper.xml'

    @autoargs.arg('alive_coeff', type=float,
                  help='reward coefficient for being alive')
    @autoargs.arg('forward_coeff', type=float,
                  help='reward coefficient for forward progress')
    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            alive_coeff=0,
            forward_coeff=1,
            ctrl_cost_coeff=0,
            *args, **kwargs):
        self.alive_coeff = alive_coeff
        self.forward_coeff = forward_coeff
        self.ctrl_cost_coeff = ctrl_cost_coeff
        super(HopperMDP, self).__init__(*args, **kwargs)
        Serializable.__init__(
            self,
            alive_coeff=alive_coeff,
            forward_coeff=forward_coeff,
            ctrl_cost_coeff=ctrl_cost_coeff,
            *args, **kwargs)

    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos[0:1],
            self.model.data.qpos[2:],
            np.clip(self.model.data.qvel, -10, 10),
            np.clip(self.model.data.qfrc_constraint, -10, 10)]
            self.get_body_com("torso"),
        ).reshape(-1)

    @overrides
    def step(self, state, action):
        next_state = self.forward_dynamics(state, action, restore=False)
        next_obs = self.get_obs(next_state)
        posbefore = state[1]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        posafter = next_state[1]
        reward = (posafter - posbefore) / self.timestep * self.forward_coeff + \
            self.alive_coeff - 0.5 * self.ctrl_cost_coeff * np.sum(np.square(action / scaling))
        notdone = np.isfinite(state).all() and \
            (np.abs(state[3:]) < 100).all() and (state[0] > .7) and \
            (abs(state[2]) < .2)
        done = not notdone
        self.state = next_state
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
    def log_extra(self):#, logger, paths):
        #forward_progress = \
        #    [path["states"][-1][1] - path["states"][0][1] for path in paths]
        forward_progress = np.concatenate(parallel_sampler.run_map(worker_collect_stats))
        #logger.record_tabular('AveragePolicyStd', np.mean(stds))

        logger.record_tabular(
            'AverageForwardProgress', np.mean(forward_progress))
        logger.record_tabular(
            'MaxForwardProgress', np.max(forward_progress))
        logger.record_tabular(
            'MinForwardProgress', np.min(forward_progress))
        logger.record_tabular(
            'StdForwardProgress', np.std(forward_progress))
