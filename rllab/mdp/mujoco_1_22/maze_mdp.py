from .mujoco_mdp import MujocoMDP
import numpy as np
from rllab.core.serializable import Serializable


class MazeMDP(MujocoMDP, Serializable):

    FILE = 'swimmer_maze.xml.mako'

    def __init__(self, *args, **kwargs):
        super(MazeMDP, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        qpos = self.model.data.qpos.flatten()[1:]
        qvel = self.model.data.qvel.flatten()
        return np.concatenate([qpos, qvel])

    def step(self, state, action):
        self.set_state(state)
        self.model.forward()
        before_com = self.get_body_com("front")
        next_state = self.forward_dynamics(state, action, restore=False)
        self.model.forward()
        after_com = self.get_body_com("front")

        next_obs = self.get_current_obs()
        ctrl_cost = 0#1e-5 * np.sum(np.square(action / 50))
        #run_cost = -1 * (after_com[0] - before_com[0])#self.model.data.qvel[0]
        run_cost = -1 * self.model.data.qvel[0]
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False
        return next_state, next_obs, reward, done
