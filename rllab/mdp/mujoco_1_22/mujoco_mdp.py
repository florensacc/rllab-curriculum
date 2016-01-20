import os
import numpy as np
from contextlib import contextmanager
import os.path as osp
import sys
import random
from rllab.mdp.base import ControlMDP
from rllab.mjcapi.rocky_mjc_1_22 import MjModel, MjViewer
from rllab.misc.overrides import overrides
import theano

class MujocoMDP(ControlMDP):

    def __init__(self, model_path, frame_skip, ctrl_scaling):
        self.model_path = model_path
        self.model = MjModel(model_path)
        self.data = self.model.data
        self.viewer = None
        self.init_qpos = self.model.data.qpos
        self.init_qvel = self.model.data.qvel
        self.init_qacc = self.model.data.qacc
        self.init_ctrl = self.model.data.ctrl
        self.qpos_dim = self.init_qpos.size
        self.qvel_dim = self.init_qvel.size
        self.ctrl_dim = self.init_ctrl.size
        self.frame_skip = frame_skip
        self.ctrl_scaling = ctrl_scaling
        self.reset()
        super(MujocoMDP, self).__init__()

    def model_path(self, file_name):
        return osp.abspath(osp.join(osp.dirname(__file__), '../../../vendor/mujoco_models/1_22/%s' % file_name))

    @property
    @overrides
    def observation_shape(self):
        return self.get_current_obs().shape

    @property
    @overrides
    def observation_dtype(self):
        return theano.config.floatX

    @property
    @overrides
    def action_dim(self):
        return len(self.model.data.ctrl)

    @property
    @overrides
    def action_dtype(self):
        return theano.config.floatX

    @property
    @overrides
    def action_bounds(self):
        bounds = self.model.actuator_ctrlrange / self.ctrl_scaling
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        return lb, ub

    @overrides
    def reset(self):
        self.model.data.qpos = self.init_qpos
        self.model.data.qvel = self.init_qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl
        self.model.forward()
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

    def forward_dynamics(self, state, action, restore=True):
        with self.set_state_tmp(state, restore):
            self.model.data.ctrl = action * self.ctrl_scaling
            for _ in range(self.frame_skip):
                self.model.step()
            self.model.forward()
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

    def set_state(self, state):
        qpos, qvel = self.decode_state(state)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model.forward()
        self.current_state = state

    @contextmanager
    def set_state_tmp(self, state, restore=True):
        if np.array_equal(state, self.current_state) and not restore:
            yield
        else:
            if restore:
                prev_pos = self.model.data.qpos
                prev_qvel = self.model.data.qvel
                prev_ctrl = self.model.data.ctrl
                prev_act = self.model.data.act
            qpos, qvel = self.decode_state(state)
            self.model.data.qpos = qpos
            self.model.data.qvel = qvel
            self.model.forward()
            yield
            if restore:
                self.model.data.qpos = prev_pos
                self.model.data.qvel = prev_qvel
                self.model.data.ctrl = prev_ctrl
                self.model.data.act = prev_act
                self.model.forward()

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def print_stats(self):
        super(MujocoMDP, self).print_stats()
        print "qpos dim:\t%d" % len(self.model.data.qpos)
