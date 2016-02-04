from .mujoco_mdp import MujocoMDP
import numpy as np
from rllab.core.serializable import Serializable


class SwimmerMazeMDP(MujocoMDP, Serializable):

    FILE = 'swimmer_maze.xml.mako'

    def __init__(self, *args, **kwargs):
        super(SwimmerMazeMDP, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        if "goal_range" in self.model.numeric_names:
            goal_range_id = self.model.numeric_names.index("goal_range")
            addr = self.model.numeric_adr.flat[goal_range_id]
            size = self.model.numeric_size.flat[goal_range_id]
            goal_range = self.model.numeric_data.flat[addr:addr+size]
            self.minx, self.maxx, self.miny, self.maxy = goal_range
        else:
            raise "Maze should have a goal range defined"

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def step(self, state, action):
        next_state = self.forward_dynamics(state, action, restore=False)
        next_obs = self.get_current_obs()
        x, y = self.get_body_com("torso")[:2]
        done = False
        reward = 0
        if self.minx <= x and x <= self.maxx and self.miny <= y \
                and y <= self.maxy:
            done = True
            reward = 1
        return next_state, next_obs, reward, done
