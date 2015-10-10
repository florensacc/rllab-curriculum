from base import MDP
import os
from mjpy import MjModel, MjViewer
import numpy as np
from contextlib import contextmanager
import os.path as osp
import sys
import random


class MjcMDP(MDP):

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = MjModel(model_path)
        self.data = self.model.data
        self.viewer = None
        self.init_qpos = self.model.data.qpos
        self.init_qvel = self.model.data.qvel
        self.init_ctrl = self.model.data.ctrl
        self.qpos_dim = self.init_qpos.size
        self.qvel_dim = self.init_qvel.size
        self.ctrl_dim = self.init_ctrl.size

    @property
    def observation_shape(self):
        raise NotImplementedError

    @property
    def n_actions(self):
        raise NotImplementedError

    def sample_initial_states(self, n):
        self.set_state(self.init_qpos, self.init_qvel)
        state = self.get_current_state()
        ob = self.get_current_obs()
        states = np.tile(state.reshape(1, -1), (n, 1))
        obs = np.tile(ob.reshape(1, -1), (n, 1))
        return states, obs

    def set_state(self, pos, vel):
        self.model.data.qpos = pos
        self.model.data.qvel = vel
        self.sig = np.random.rand()
        self.model.forward()

    def get_state(self, sig, pos, vel):
        return np.concatenate([pos.reshape(-1), vel.reshape(-1), [sig]])

    def get_current_obs(self):
        raise NotImplementedError

    def get_obs(self, state):
        with self.set_state_tmp(state):
            return self.get_current_obs()

    def get_current_state(self):
        return self.get_state(self.sig, self.model.data.qpos, self.model.data.qvel)

    def step(self, states, actions):
        raise NotImplementedError

    def get_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer()
            self.viewer.start()
            self.viewer.set_model(self.model)
        return self.viewer

    def plot(self):
        viewer = self.get_viewer()
        viewer.loop_once()

    def start_viewer(self):
        viewer = self.get_viewer()
        if not viewer.running:
            viewer.start()

    def stop_viewer(self):
        if self.viewer:
            self.viewer.finish()

    @contextmanager
    def set_state_tmp(self, state):
        qpos, qvel, sig = np.split(state, [self.qpos_dim, self.qpos_dim+self.qvel_dim])
        sig = sig[0]
        if sig == self.sig:
            yield
        else:
            prev_pos = self.model.data.qpos
            prev_qvel = self.model.data.qvel
            prev_ctrl = self.model.data.ctrl
            self.model.data.qpos = qpos
            self.model.data.qvel = qvel
            self.model.forward()
            yield
            self.model.data.qpos = prev_pos
            self.model.data.qvel = prev_qvel
            self.model.data.ctrl = prev_ctrl
            self.model.forward()

class HopperMDP(MjcMDP):
    def __init__(self):
        self.frame_skip = 5
        self.ctrl_scaling = 100.0
        self.timestep = .02
        MjcMDP.__init__(self, osp.abspath(osp.join(osp.dirname(__file__), '../vendor/mujoco_models/hopper.xml')))

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos[0:1],
            self.model.data.qpos[2:],
            np.clip(self.model.data.qvel,-10,10),
            np.clip(self.model.data.qfrc_constraint,-10,10)]
        ).reshape(-1)

    @property
    def observation_shape(self):
        return self.sample_initial_state()[1].shape

    @property
    def n_actions(self):
        return len(self.model.data.ctrl)

    def step(self, states, actions):
        assert len(states) == 1

        state = states[0]
        action = actions[0]

        with self.set_state_tmp(state):
            posbefore = self.model.data.qpos[1,0]
            self.model.data.ctrl = action * self.ctrl_scaling

            for _ in range(self.frame_skip):
                self.model.step()

            posafter = self.model.data.qpos[1,0]
            reward = (posafter - posbefore) / self.timestep + 2.0## + 3.0
            state = self.get_current_state()
            ob = self.get_current_obs()
            notdone = np.isfinite(state).all() and (np.abs(state[3:])<100).all() and (state[0] > .7) and (abs(state[2]) < .2)
            done = not notdone

        if done:
            state, ob = self.sample_initial_state()
        return [state], [ob], [reward], [done], [1]

#class WalkerMDP(MjcMDP):
#
#    def __init__(self):
#        self.frame_skip = 4
#        self.ctrl_scaling = 20.0
#        self.timestep = .02
#        MjcMDP.__init__(self, osp.abspath(osp.join(osp.dirname(__file__), '../vendor/mujoco_models/walker2d.xml')))
#
#    def get_obs(self, state):
#        with self.set_state_tmp(state):
#            return np.concatenate([self.model.data.qpos, np.sign(self.model.data.qvel), np.sign(self.model.data.qfrc_constraint)]).reshape(1,-1)
#
#    @property
#    def observation_shape(self):
#        return self.sample_initial_state()[1].shape
#
#    @property
#    def n_actions(self):
#        return len(self.model.data.ctrl)
#
#    def step(self, a):
#
#        posbefore = self.model.data.xpos[:,0].min()
#        self.model.data.ctrl = a * self.ctrl_scaling
#
#        for _ in range(self.frame_skip):
#            self.model.step()
#
#        posafter = self.model.data.xpos[:,0].min()
#        reward = (posafter - posbefore) / self.timestep + 1.0
#
#        s = np.concatenate([self.model.data.qpos, self.model.data.qvel])
#        notdone = np.isfinite(s).all() and (np.abs(s[3:])<100).all() and (s[0] > 0.7) and (abs(s[2]) < .5)
#        done = not notdone
#
#        ob = self._get_obs()
#
#        return ob, reward, done
