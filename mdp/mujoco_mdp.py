from .base import ControlMDP
import os
from mjpy import MjModel, MjViewer
import numpy as np
from contextlib import contextmanager
import os.path as osp
import sys
import random
from misc.overrides import overrides

class MujocoMDP(ControlMDP):

    def __init__(self, model_path, horizon, frame_skip, ctrl_scaling):
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
        self.frame_skip = frame_skip
        self.ctrl_scaling = ctrl_scaling
        self.reset()
        super(MujocoMDP, self).__init__(horizon)

    @property
    @overrides
    def observation_shape(self):
        return self.get_current_obs().shape

    @property
    @overrides
    def n_actions(self):
        return len(self.model.data.ctrl)

    @property
    @overrides
    def action_dtype(self):
        return theano.config.floatX

    @overrides
    def reset(self):
        self.model.data.qpos = self.init_qpos
        self.model.data.qvel = self.init_qvel
        self.model.data.ctrl = self.init_ctrl
        self.current_state = self.get_current_state()
        return self.get_current_state(), self.get_current_obs()

    def get_state(self, pos, vel):
        return np.concatenate([pos.reshape(-1), vel.reshape(-1)])

    def decode_state(self, state):
        qpos, qvel = np.split(state, [self.qpos_dim])
        #qvel = state[self.qpos_dim:self.qpos_dim+self.qvel_dim]
        return qpos, qvel

    def get_current_obs(self):
        raise NotImplementedError

    def get_obs(self, state):
        with self.set_state_tmp(state):
            return self.get_current_obs()

    def get_current_state(self):
        return self.get_state(self.model.data.qpos, self.model.data.qvel)

    def forward_dynamics(self, state, action, preserve=True):
        with self.set_state_tmp(state, preserve):
            self.model.data.ctrl = action * self.ctrl_scaling
            for _ in range(self.frame_skip):
                self.model.step()
            #self.model.forward()
            return self.get_current_state()

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
    def set_state_tmp(self, state, preserve=True):
        if np.array_equal(state, self.current_state) and not preserve:
            yield
        else:
            if preserve:
                prev_pos = self.model.data.qpos
                prev_qvel = self.model.data.qvel
                prev_ctrl = self.model.data.ctrl
                prev_act = self.model.data.act
            qpos, qvel = self.decode_state(state)
            self.model.data.qpos = qpos
            self.model.data.qvel = qvel
            self.model.forward()
            yield
            if preserve:
                self.model.data.qpos = prev_pos
                self.model.data.qvel = prev_qvel
                self.model.data.ctrl = prev_ctrl
                self.model.data.act = prev_act
                self.model.forward()
