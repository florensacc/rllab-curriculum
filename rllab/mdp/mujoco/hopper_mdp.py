from rllab.mdp.mujoco_mdp import MujocoMDP
import numpy as np
from rllab.core.serializable import Serializable

# states: [
# 0: z-coord,
# 1: x-coord (forward distance),
# 2: forward pitch along y-axis,
# 6: z-vel (up = +),
# 7: xvel (forward = +)
class HopperMDP(MujocoMDP, Serializable):
    def __init__(self, horizon=1000, timestep=0.02):
        frame_skip = 1#5#10#15#5#1#5#25#10##5
        ctrl_scaling = 100.0
        self.timestep = timestep
        path = self.model_path('hopper.xml')
        super(HopperMDP, self).__init__(path, horizon, frame_skip, ctrl_scaling)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos[0:1],
            self.model.data.qpos[2:],
            np.clip(self.model.data.qvel,-10,10),
            np.clip(self.model.data.qfrc_constraint,-10,10)]
        ).reshape(-1)

    def step(self, state, action, autoreset=True):
        next_state = self.forward_dynamics(state, action, restore=False)
        next_obs = self.get_obs(next_state)
        posbefore = state[1]
        posafter = next_state[1]
        reward = (posafter - posbefore) / self.timestep# + 3.0
        notdone = np.isfinite(state).all() and (np.abs(state[3:])<100).all() and (state[0] > .7) and (abs(state[2]) < .2)
        done = not notdone
        #if done:
        #    next_state, next_obs = self.reset()
        #else:
        self.state = next_state
        return next_state, next_obs, reward, done

    def cost(self, state, action):
        return 0.001*sum(np.square(action-1))# + 10*(state[7] - 0.1)**2#/ self.timestep - 1.5)# + 1*np.square(state[0] - 1.25)
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
        lb[3:] = -100
        ub[3:] = 100
        lb[0] = 0.7
        lb[2] = -0.2
        ub[2] = 0.2
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
    
    def demo_policy(self, policy):
        self.start_viewer()
        self.viewer.clear_frames()
        s, o = self.reset()
        while True:
            a,_ = policy.get_action(o)
            s, o, r, d = self.step(s, a)#, autoreset=False)
            if d:
                s, o = self.reset()
            #if idx % 10 == 0:
            #self.viewer.record_frame(alpha=0.5)
            self.viewer.loop_once()
            import time
            time.sleep(0.02)
        while True:
            #self.model.step()
            self.viewer.loop_once()
    def demo(self, actions):

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
            import time
            time.sleep(0.02)
        while True:
            #self.model.step()
            self.viewer.loop_once()
