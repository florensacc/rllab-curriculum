from rllab.mdp.mujoco_1_20.mujoco_mdp import MujocoMDP
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
from rllab.misc.overrides import overrides

# states: [
# 0: z-coord,
# 1: x-coord (forward distance),
# 2: forward pitch along y-axis,
# 6: z-vel (up = +),
# 7: xvel (forward = +)


class HopperMDP(MujocoMDP, Serializable):

    @autoargs.arg('alive_coeff', type=float,
                  help='reward coefficient for being alive')
    @autoargs.arg('forward_coeff', type=float,
                  help='reward coefficient for forward progress')
    def __init__(
            self,
            alive_coeff=0,
            forward_coeff=1):
        self.alive_coeff = alive_coeff
        self.forward_coeff = forward_coeff
        self.state = None
        path = self.model_path('hopper.xml')
        super(HopperMDP, self).__init__(path, frame_skip=1, ctrl_scaling=1)
        Serializable.__init__(self, alive_coeff, forward_coeff)

    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos[0:1],
            self.model.data.qpos[2:],
            np.clip(self.model.data.qvel, -10, 10),
            np.clip(self.model.data.qfrc_constraint, -10, 10)]
        ).reshape(-1)

    @overrides
    def step(self, state, action):
        next_state = self.forward_dynamics(state, action, restore=False)
        next_obs = self.get_obs(next_state)
        posbefore = state[1]
        posafter = next_state[1]
        reward = (posafter - posbefore) / self.model.opt.timestep * self.forward_coeff + \
            self.alive_coeff
        notdone = np.isfinite(state).all() and \
            (np.abs(state[3:]) < 100).all() and (state[0] > .7) and \
            (abs(state[2]) < .2)
        done = not notdone
        self.state = next_state
        return next_state, next_obs, reward, done

    @overrides
    def log_extra(self, logger, paths):
        forward_progress = \
            [path["states"][-1][1] - path["states"][0][1] for path in paths]
        logger.record_tabular(
            'AverageForwardProgress', np.mean(forward_progress))
        logger.record_tabular(
            'MaxForwardProgress', np.max(forward_progress))
        logger.record_tabular(
            'MinForwardProgress', np.min(forward_progress))
        logger.record_tabular(
            'StdForwardProgress', np.std(forward_progress))
