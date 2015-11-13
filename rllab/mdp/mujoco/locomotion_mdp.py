from .mujoco_mdp import MujocoMDP
import os
import numpy as np
from contextlib import contextmanager
import os.path as osp
import sys
import random

# DOFs:
# 0: forward (y-axis) distance
# 1: z-axis height
# 2: rotation around x-axis (rotation)
# 3: rotation around z-axis (turn)
# 4: rotation around y-axis (shake)
# 5: x-axis distance
# 6: right thigh rotation around x axis (up is positive)
# 7: right thigh rotation around y axis (left is positive)
# 8: right calf rotation around x axis (up is positive)
# 9: right foot rotation around x axis (up is positive)
# 10: left thigh rotation around x axis (up is positive)
# 11: left thigh rotation around y axis (left is positive)
# 12: left calf rotation around x axis (up is positive)
# 13: left foot rotation around x axis (up is positive)
class LocomotionMDP(MujocoMDP):
    def __init__(self, horizon=50, timestep=0.04):
        frame_skip = 20#10#15#5#1#5#25#10##5
        ctrl_scaling = 20.0
        self.timestep = timestep
        path = osp.abspath(osp.join(osp.dirname(__file__), '../vendor/mujoco_models/locomotion.xml'))
        super(LocomotionMDP, self).__init__(path, horizon, frame_skip, ctrl_scaling)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos[1:],
            np.clip(self.model.data.qvel,-10,10),
            np.clip(self.model.data.qfrc_constraint,-10,10)]
        ).reshape(-1)

    def step(self, state, action, autoreset=True):
        next_state = self.forward_dynamics(state, action, preserve=False)
        next_obs = self.get_obs(next_state)
        posbefore = state[0]
        posafter = next_state[0]
        reward = (posafter - posbefore) / self.timestep + 1.0
        notdone = np.isfinite(state).all() \
                and np.abs(state[2]) < .2 \
                and np.abs(state[4]) < .2 \
                and state[0] > -0.1 \
                and np.abs(state[6]) < 1 \
                and np.abs(state[7]) < 1 \
                and np.abs(state[10]) < 1 \
                and np.abs(state[11]) < 1 \
                and np.abs(state[1]) < .05
        #notdone = np.isfinite(state).all() and (np.abs(state[3:])<100).all() and (state[0] > .7) and (abs(state[2]) < .2)
        #done = not notdone
        #if done:
        #    next_state, next_obs = self.reset()
        #else:
        done = not notdone
        self.state = next_state
        return next_state, next_obs, reward, done

    def cost(self, state, action):
        return 0.01*sum(np.square(action)) + np.square(state[14]-0.1)#-1))# + 10*(state[7] - 0.1)**2#/ self.timestep - 1.5)# + 1*np.square(state[0] - 1.25)
        #if np.any(state[6:] != 0):
        #    with self.set_state_tmp(state):
        #        import ipdb; ipdb.set_trace()
        #        return 0
        #return 0

    def final_cost(self, state):
        return 0#0.5*np.square(state[1] - 15)

    @property
    def state_bounds(self):
        s = self.get_current_state()
        lb = np.ones_like(s) * -np.inf#-1000#-np.inf
        ub = np.ones_like(s) * np.inf#1000#np.inf
        #lb[3:] = -100
        #ub[3:] = 100
        #lb[0] = 0.7
        #lb[2] = -0.2
        #ub[2] = 0.2
        return lb, ub

    # initial trajectory, used for colocation 
    @property
    def xinit(self):
        start_pos = self.init_qpos
        start_vel = self.init_qvel
        start_state = np.concatenate([start_pos, start_vel]).reshape(-1)
        end_pos = np.array(start_pos)
        end_pos[1, 0] = 5#10#0, 1] = 10
        end_vel = self.init_qvel
        end_state = np.concatenate([end_pos, end_vel]).reshape(-1)
        xs = np.array([np.linspace(s, e, num=self.horizon+1) for s, e in zip(start_state, end_state)]).T
        return xs
        #qpos = np.array(self.model.data.qpos)
        #qpos[0,0] = 10
        #from mjpy.mjlib import mjlib
        #self.model.data.qpos = qpos
        #print self.model.data.qpos
        #print self.model.data.xpos
        #self.model.forward()
        #print self.model.data.qpos
        #print self.model.data.xpos
        #import ipdb; ipdb.set_trace()

    def demo(self, actions, exit_when_done=False):

        self.start_viewer()
        #while True:
        self.viewer.clear_frames()
        #self.model.data.qpos = np.array([[1.57, 1.57]])
        #self.viewer.record_frame(emission=100, alpha=1)
        s, _ = self.reset()
        #with self.set_state_tmp(np.array([1.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])):
            #self.viewer.record_frame(emission=100, alpha=1)
        for idx, a in enumerate(actions):
            s, _, _, _ = self.step(s, a)#, autoreset=False)
            #if idx % 10 == 0:
            #self.viewer.record_frame(alpha=0.5)
            self.viewer.loop_once()
            #import time
            #time.sleep(0.02)
        if not exit_when_done:
            while True:
                #self.model.step()
                self.viewer.loop_once()
