from .mujoco_mdp import MujocoMDP
import numpy as np
from core.serializable import Serializable


class SwimmerMDP(MujocoMDP, Serializable):

    def __init__(self, horizon=500, timestep=0.02):
        self.timestep = timestep
        frame_skip = 25
        path = self.model_path('swimmer.xml')
        ctrl_scaling = 30
        super(SwimmerMDP, self).__init__(path, horizon, frame_skip, ctrl_scaling)
        Serializable.__init__(self, horizon, timestep)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos[1:],
            np.clip(self.model.data.qvel,-10,10),
            np.clip(self.model.data.qfrc_constraint,-10,10),
        ]).reshape(-1)

    def cost(self, state, action):
        #with self.set_state_tmp(state):
        #    self.model.data.xipos[1:]
        #    return np.square(self.model.data.cvel[1][3] - .1) + 1e-6*np.sum(np.square(action))
        #com_before = self.data.com_subtree[0]
        #print state
        #import ipdb; ipdb.set_trace()
        return np.square(state[5] / self.timestep - 1) + 1e-5*np.sum(np.square(action))

    def final_cost(self, state):
        return 0
        #with self.set_state_tmp(state):
        #    return np.square(self.model.data.cvel[1][3] - .1)# + 1e-6*np.sum(np.square(action))

        #return np.square(state[5] - .1)
        #return 0

    @property
    def state_bounds(self):
        s = self.get_current_state()
        lb = np.ones_like(s) * -np.inf#-1000#-np.inf
        ub = np.ones_like(s) * np.inf#1000#np.inf
        return lb, ub

    def step(self, state, action):
        #com_before = self.data.com_subtree[0]
        next_state = self.forward_dynamics(state, action, preserve=False)
        #com_after = self.data.com_subtree[0]
        #import ipdb; ipdb.set_trace()
        
        #next_obs = self.get_obs(next_state)
        #posbefore = state[0]
        #posafter = next_state[0]
        #notdone = np.isfinite(state).all() \
        #        and np.abs(state[2]) < .2 \
        #        and np.abs(state[4]) < .2 \
        #        and state[0] > -0.1 \
        #        and np.abs(state[6]) < 1 \
        #        and np.abs(state[7]) < 1 \
        #        and np.abs(state[10]) < 1 \
        #        and np.abs(state[11]) < 1 \
        #        and np.abs(state[1]) < .05
        #notdone = np.isfinite(state).all() and (np.abs(state[3:])<100).all() and (state[0] > .7) and (abs(state[2]) < .2)
        #done = not notdone
        #if done:
        #    next_state, next_obs = self.reset()
        #else:
        #done = not notdone
        self.state = next_state
        #with self.set_state_tmp
        reward = self.state[5] / self.timestep - 1e-5*np.sum(np.square(action))#state[5] / self.timestep#(posafter - posbefore) / self.timestep# + 1.0
        return next_state, self.get_obs(next_state), reward, False


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
