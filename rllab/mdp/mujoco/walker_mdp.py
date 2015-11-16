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
            self.model.data.qpos,
            np.sign(self.model.data.qvel),
            np.sign(self.model.data.qfrc_constraint),
        ]).reshape(-1)

    def step(self, state, action):
        self.set_state(state)#state = next_state
        posbefore = self.model.data.xpos[:,0].min()
        next_state = self.forward_dynamics(state, action, preserve=False)
        self.current_state = next_state
        #reward = 

        #self.model.data.ctrl = a * self.ctrl_scaling

        #for _ in range(self.frame_skip):
        #    self.model.step()

        posafter = self.model.data.xpos[:,0].min()
        reward = (posafter - posbefore) / self.timestep + 1.0

        #s = np.concatenate([self.model.data.qpos, self.model.data.qvel])
        notdone = np.isfinite(next_state).all() and (np.abs(next_state[3:])<100).all() and (next_state[0] > 0.7) and (abs(next_state[2]) < .5)
        done = not notdone

        ob = self.get_current_obs()

        return next_state, ob, reward, done
