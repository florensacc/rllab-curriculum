from mujoco_mdp import MujocoMDP
from rllab.core.serializable import Serializable
import numpy as np

class AntMDP(MujocoMDP, Serializable):

    def __init__(self, horizon=250, timestep=0.01):
        frame_skip = 1
        ctrl_scaling = 100.0
        self.timestep = timestep
        path = self.model_path('ant.xml')
        super(AntMDP, self).__init__(path, horizon, frame_skip, ctrl_scaling)
        Serializable.__init__(self, horizon, timestep)
        self.init_qpos = np.array([0., 0., 0.55, 1., 0., 0., 0., 0., 1.0, 0., -1.0, 0., -1.0, 0., 1.0])

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos[:1],
            self.model.data.qpos[2:],
            np.sign(self.model.data.qvel),
            np.sign(self.model.data.qfrc_constraint),
        ]).reshape(-1)

    def get_current_com(self):
        xipos = self.model.data.xipos[1:]
        body_mass = self.model.body_mass[1:]
        return (xipos * body_mass).sum(axis=0) / body_mass.sum()

    def step(self, state, action):
        self.set_state(state)
        #com_before = self.get_current_com()
        next_state = self.forward_dynamics(state, action, restore=False)
        #com_after = self.get_current_com()
        #reward = (com_after[0] - com_before[0]) / self.timestep + 1.0
        posbefore = state[0]
        posafter = next_state[0]
        reward = (posafter - posbefore) / self.timestep + 1.0# - 1e-4*np.sum(np.square(action))
        notdone = np.isfinite(next_state).all() and next_state[2] >= 0.2 and next_state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return next_state, ob, reward, done

if __name__ == "__main__":
    mdp = AntMDP()
    state = mdp.reset()[0]
    mdp.start_viewer()
    mdp.plot()
    while True:
        state = mdp.step(state, np.zeros(mdp.action_dim))[0]
        mdp.viewer.loop_once()

    print mdp.calc_current_com()
