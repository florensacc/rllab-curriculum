from .mujoco_mdp import MujocoMDP
import os
import numpy as np
from contextlib import contextmanager
import os.path as osp
import sys
import random

class GripperMDP(MujocoMDP):
    def __init__(self, horizon=100):
        frame_skip = 10#10#40
        ctrl_scaling = 100#100.0
        path = osp.abspath(osp.join(osp.dirname(__file__), '../vendor/mujoco_models/gripper.xml'))
        super(GripperMDP, self).__init__(path, horizon, frame_skip, ctrl_scaling)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos,
            self.model.data.qvel,
        ]).reshape(-1)

    def step(self, state, action, autoreset=True):
        next_state = self.forward_dynamics(state, action, preserve=False)
        next_obs = self.get_obs(next_state)
        #posbefore = state[1]
        #posafter = next_state[1]
        reward = 0#(posafter - posbefore) / self.timestep + 3.0
        notdone = False#np.isfinite(state).all() and (np.abs(state[3:])<100).all() and (state[0] > .7) and (abs(state[2]) < .2)
        done = not notdone
        self.state = next_state
        return next_state, next_obs, reward, done


    @property
    def xinit(self):
        x0 = np.array([0, 0, 0, 0])#, _ = self.reset()
        xf = np.array([1.57, 1.57, 0, 0])
        return np.array([np.linspace(s, e, self.horizon+1) for s, e in zip(x0, xf)]).T

    def cost(self, state, action):
        return 0.01*np.square(action).sum()

    def final_cost(self, state):
        return 50*np.square(state[0:2] - np.array([1.57, 1.57])).sum()

    def demo(self, actions):

        self.start_viewer()
        #while True:
        self.viewer.clear_frames()
        self.model.data.qpos = np.array([[1.57, 1.57]])
        self.viewer.record_frame(emission=100, alpha=1)
        s, _ = self.reset()
        self.viewer.record_frame(emission=100, alpha=1)
        for idx, a in enumerate(actions):
            s, _, _, _ = self.step(s, a)#, autoreset=False)
            if idx % 10 == 0:
                self.viewer.record_frame(alpha=0.5)
            self.viewer.loop_once()
            import time
            time.sleep(0.02)
        while True:
            self.viewer.loop_once()
