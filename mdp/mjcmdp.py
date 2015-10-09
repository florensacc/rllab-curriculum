from base import MDP#, Serializable
import os
from mjpy import MjModel, MjViewer
import numpy as np

class MjcMDP(MDP):

    def __init__(self):
        self.model = MjModel(self.model_path())
        self.data = self.model.data
        self.viewer = None
        self.init_qpos = self.model.data.qpos
        self.init_qvel = self.model.data.qvel
        self.ctrl_dim = self.model.data.ctrl.size

        o,r,_ = self.step(np.zeros(self.ctrl_dim))
        self.obs_dim = o.size
        self.rew_dim = r.size

    def model_path(self):
        raise NotImplementedError

    def step(self, s, a):
        raise NotImplementedError

    def get_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer()
            self.viewer.start()
            self.viewer.set_model(self.model)
        return self.viewer

    def action_spec(self):
        return (np.float64,(self.model.nu,))

    def observation_spec(self):
        return (np.float64, (self.obs_dim,))

    def plot(self):
        viewer = self.get_viewer()
        viewer.loop_once()


class HopperMDP(MjcMDP):#, Serializable):
    def __init__(self):
        self.frame_skip = 5
        self.ctrl_scaling = 100.0
        self.timestep = .02
        MjcMDP.__init__(self)
        #Serializable.__init__(self)

    def model_path(self):
        return os.path.join(os.path.dirname(__file__),'../vendor/mujoco_models/hopper.xml')
    
    def reset(self):
        self.x0 = np.concatenate([self.model.data.qpos, self.model.data.qvel])
        self.model.data.qpos = self.init_qpos
        self.model.data.qvel = self.init_qvel
        # self.model.data.qvel = np.random.randn(len(self.model.data.qvel))*.2
        # self.model.data.qfrc_constraint[:] = 0
        self.model.forward()
        return self._get_obs()

    def _get_obs(self):
        qpos = self.model.data.qpos
        return np.concatenate([qpos[0:1], qpos[2:], np.clip(self.model.data.qvel,-10,10), np.clip(self.model.data.qfrc_constraint,-10,10)]).reshape(1,-1)

    @property
    def observation_shape(self):
        return self._get_obs().shape

    @property
    def n_actions(self):
        return len(self.model.data.ctrl)

    def step(self, a):

        posbefore = self.model.data.qpos[1]
        self.model.data.ctrl = a * self.ctrl_scaling

        for _ in range(self.frame_skip):
            self.model.step()

        posafter = self.model.data.qpos[1]
        reward = (posafter - posbefore) / self.timestep + 3.0

        s = np.concatenate([self.model.data.qpos, self.model.data.qvel])
        notdone = np.isfinite(s).all() and (np.abs(s[3:])<100).all() and (s[0] > .7) and (abs(s[2]) < .2)
        done = not notdone

        ob = self._get_obs()

        return ob, reward, done

    def reward_names(self):
        return ["vel"]

class WalkerMDP(MjcMDP):#,Serializable):
    def __init__(self):
        self.frame_skip = 4
        self.ctrl_scaling = 20.0
        self.timestep = .02
        MjcMDP.__init__(self)
        #Serializable.__init__(self)


    def model_path(self):
        return os.path.join(os.path.dirname(__file__),'vendor/mujoco_models/walker2d.xml')
    
    def reset(self):
        self.x0 = np.concatenate([self.model.data.qpos, self.model.data.qvel])
        self.model.data.qpos = self.init_qpos
        self.model.data.qvel = self.init_qvel
        # self.model.data.qvel = np.random.randn(len(self.model.data.qvel))*.2
        # self.model.data.qfrc_constraint[:] = 0
        self.model.forward()
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.model.data.qpos, np.sign(self.model.data.qvel), np.sign(self.model.data.qfrc_constraint)]).reshape(1,-1)


    def step(self, a):

        posbefore = self.model.data.xpos[:,0].min()
        self.model.data.ctrl = a * self.ctrl_scaling

        for _ in range(self.frame_skip):
            self.model.step()

        posafter = self.model.data.xpos[:,0].min()
        reward = (posafter - posbefore) / self.timestep + 1.0

        s = np.concatenate([self.model.data.qpos, self.model.data.qvel])
        notdone = np.isfinite(s).all() and (np.abs(s[3:])<100).all() and (s[0] > 0.7) and (abs(s[2]) < .5)
        done = not notdone

        ob = self._get_obs()

        return ob, reward, done

    def reward_names(self):
        return ["vel"]
