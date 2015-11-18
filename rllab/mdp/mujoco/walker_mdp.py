from mujoco_mdp import MujocoMDP
from rllab.core.serializable import Serializable
import numpy as np

class WalkerMDP(MujocoMDP, Serializable):

    def __init__(self, horizon=250, timestep=0.02):
        frame_skip = 4
        ctrl_scaling = 20.0
        self.timestep = .02
        path = self.model_path('walker2d.xml')
        super(WalkerMDP, self).__init__(path, horizon, frame_skip, ctrl_scaling)
        Serializable.__init__(self, horizon, timestep)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos[:1],
            self.model.data.qpos[2:],
            np.sign(self.model.data.qvel),
            np.sign(self.model.data.qfrc_constraint),
        ]).reshape(-1)

    def step(self, state, action):
        next_state = self.forward_dynamics(state, action, preserve=False)
        posbefore = state[1]
        posafter = next_state[1]
        reward = (posafter - posbefore) / self.timestep + 1.0
        notdone = np.isfinite(next_state).all() and (np.abs(next_state[3:])<100).all() and (next_state[0] > 0.7) and (abs(next_state[2]) < .5)
        done = not notdone
        ob = self.get_current_obs()
        return next_state, ob, reward, done
