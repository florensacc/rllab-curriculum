from rl import MDP
import os
from mjpy import MjModel, MjViewer
import numpy as np

class MjcMDP(MDP):

    def __init__(self):
        self.model = MjModel(self.model_path())
        self.data = self.model.data
        self.viewer = None
        self.x0 = self.get_state()
        self.x0.setflags(write=False)

    def get_state(self):
        return self.encode_state(self.model.data.qpos, self.model.data.qvel)

    def encode_state(self, pos, vel):
        return np.concatenate([pos, vel])

    def decode_state(self, state):
        pos = state[0:self.model.nq]
        vel = state[self.model.nq:self.model.nq+self.model.nv]
        return pos, vel

    def start_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer()
        self.viewer.start()
        self.viewer.set_model(self.model)
        return self.viewer

    @property
    def frame_skip(self):
        return 1

    def step(self, states, actions):
        next_states = []
        obs = []
        rewards = []
        dones = []
        for state, action in zip(states, actions):
            pos, vel = self.decode_state(state)
            self.model.data.qpos = pos
            self.model.data.qvel = vel

            self.model.data.ctrl = action

            for _ in range(self.frame_skip):
                self.model.step()
            next_state = self.get_state()
            ob = next_state
            reward = 0
            done = False

            next_states.append(next_state)
            obs.append(ob)
            rewards.append(reward)
            dones.append(done)

        return next_states, obs, rewards, dones

    def sample_initial_states(self, n):
        return [self.x0 for _ in range(n)], [self.x0 for _ in range(n)]

    @property
    def action_shape(self):
        return self.model.data.ctrl.shape

class WalkerMDP(MjcMDP):
    
    def __init__(self):
        super(WalkerMDP, self).__init__()

    def model_path(self):
        return os.path.join(os.path.dirname(__file__),
                              'vendor/mujoco_models/walker2d.xml')

    @property
    def frame_skip(self):
        return 5
    
class CartpoleMDP(MjcMDP):
    def __init__(self):
        super(CartpoleMDP, self).__init__()

    def model_path(self):
        return os.path.join(os.path.dirname(__file__),
                              'vendor/mujoco_models/cartpole.xml')

    @property
    def frame_skip(self):
        return 1

class HopperMDP(MjcMDP):
    def __init__(self):
        super(HopperMDP, self).__init__()

    def model_path(self):
        return os.path.join(os.path.dirname(__file__),
                              'vendor/mujoco_models/hopper.xml')

    @property
    def frame_skip(self):
        return 5

class HumanoidMDP(MjcMDP):
    def __init__(self):
        super(HumanoidMDP, self).__init__()

    def model_path(self):
        return os.path.join(os.path.dirname(__file__),
                              'vendor/mujoco_models/humanoid.xml')

    @property
    def frame_skip(self):
        return 8
