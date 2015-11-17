import polopt
import os.path as osp
from rllab.mdp.base import ControlMDP
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from contextlib import contextmanager
import numpy as np

class MujocoMDP(ControlMDP):

    def __init__(self, model_name=None, ctrl_scaling=1, frame_skip=1, model_path=None):
        mujoco = polopt.MujocoSimulator()
        if model_path is None:
            if model_name is None:
                raise ValueError("Must fill either model_name or model_path")
            mujoco.create(self.model_path(model_name))
        else:
            mujoco.create(model_path)
        self.init_qpos = mujoco.qpos
        self.init_qvel = mujoco.qvel
        self.init_ctrl = mujoco.ctrl
        self.qpos_dim = self.init_qpos.size
        self.qvel_dim = self.init_qvel.size
        self.ctrl_dim = self.init_ctrl.size
        self.mujoco = mujoco
        self.ctrl_scaling = ctrl_scaling
        self.frame_skip = frame_skip

    @property
    @overrides
    def observation_shape(self):
        return self.get_current_obs().shape

    @property
    @overrides
    def action_dim(self):
        return len(self.mujoco.ctrl)

    @property
    @overrides
    def action_dtype(self):
        return theano.config.floatX


    @overrides
    def reset(self):
        self.mujoco.qpos = self.init_qpos
        self.mujoco.qvel = self.init_qvel
        self.mujoco.ctrl = self.init_ctrl
        self.current_state = self.get_current_state()
        return self.get_current_state(), self.get_current_obs()

    def get_state(self, pos, vel):
        return np.concatenate([pos.reshape(-1), vel.reshape(-1)])

    def decode_state(self, state):
        qpos, qvel = np.split(state, [self.qpos_dim])
        return qpos, qvel

    def get_current_obs(self):
        raise NotImplementedError

    def get_obs(self, state):
        with self.set_state_tmp(state):
            return self.get_current_obs()

    def get_current_state(self):
        return self.get_state(self.mujoco.qpos, self.mujoco.qvel)

    def forward_dynamics(self, state, action, preserve=True):
        with self.set_state_tmp(state, preserve):
            self.mujoco.ctrl = action * self.ctrl_scaling
            for _ in range(self.frame_skip):
                self.mujoco.step()
            return self.get_current_state()

    def get_viewer(self):
        return
        if self.viewer is None:
            self.viewer = MjViewer()
            self.viewer.start()
            self.viewer.set_model(self.model)
        return self.viewer

    def plot(self):
        return
        viewer = self.get_viewer()
        viewer.loop_once()

    def start_viewer(self):
        return
        viewer = self.get_viewer()
        if not viewer.running:
            viewer.start()

    def stop_viewer(self):
        return
        if self.viewer:
            self.viewer.finish()

    def set_state(self, state):
        qpos, qvel = self.decode_state(state)
        self.mujoco.qpos = qpos
        self.mujoco.qvel = qvel
        self.mujoco.forward()
        self.current_state = state

    @contextmanager
    def set_state_tmp(self, state, preserve=True):
        if np.array_equal(state, self.current_state) and not preserve:
            yield
        else:
            if preserve:
                prev_qpos = self.mujoco.qpos
                prev_qvel = self.mujoco.qvel
                prev_ctrl = self.mujoco.ctrl
            qpos, qvel = self.decode_state(state)
            self.mujoco.qpos = qpos
            self.mujoco.qvel = qvel
            self.mujoco.forward()
            yield
            if preserve:
                self.mujoco.qpos = prev_qpos
                self.mujoco.qvel = prev_qvel
                self.mujoco.ctrl = prev_ctrl
                self.mujoco.forward()

    def model_path(self, file_name):
        return osp.abspath(osp.join(osp.dirname(__file__), '../../vendor/igor_mjc/%s' % file_name))

class AcrobotMDP(MujocoMDP, Serializable):

    def __init__(self, ctrl_scaling=100):
        super(AcrobotMDP, self).__init__("acrobot.xml", ctrl_scaling=ctrl_scaling)
        self.init_qpos = np.array([np.pi, 0])
        Serializable.__init__(self, ctrl_scaling)

    def get_current_obs(self):
        return np.concatenate([
                self.mujoco.qpos,
                self.mujoco.qvel,
        ]).reshape(-1)

    def step(self, state, action):
        next_state = self.forward_dynamics(state, action, preserve=False)
        self.current_state = next_state
        reward = -np.sum(np.square(state)) - 1e-5*np.sum(np.square(action))## self.state[5] / self.timestep - 1e-5*np.sum(np.square(action))#state[5] / self.timestep#(posafter - posbefore) / self.timestep# + 1.0
        return next_state, self.get_obs(next_state), reward, False


if __name__ == "__main__":
    mdp = AcrobotMDP()
    #mdp.init_qpos = np.array([np.pi, 0])

    state = mdp.reset()[0]
    print "initial state:", state
    print "control dim:", mdp.action_dim

    states = [state]
    for _ in range(100):
        state = mdp.step(state, np.zeros(1))[0]
        print state
        states.append(state)
    #np.
    #state = np.array([np.pi/2, np.pi/2])
    #states = np.copy(np.array([np.linspace(0, np.pi, 100),np.linspace(0, np.pi, 100)]).T)
    #polopt.replayTrajectory(mdp.mujoco, np.array([state]), False, False)
    #print "start..."
    polopt.replayTrajectory(mdp.mujoco, np.array(states), True, False)
    #print mdp.qpos_dim
