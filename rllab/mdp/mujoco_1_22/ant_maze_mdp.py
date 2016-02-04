from .mujoco_mdp import MujocoMDP
import numpy as np
from rllab.core.serializable import Serializable


class AntMazeMDP(MujocoMDP, Serializable):

    FILE = 'ant_maze.xml.mako'

    def __init__(self, *args, **kwargs):
        super(AntMazeMDP, self).__init__(*args, **kwargs)
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
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

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
        notdone = np.isfinite(next_state).all() \
            and next_state[2] >= 0.2 and next_state[2] <= 1.0
        done = done or (not notdone)
        return next_state, next_obs, reward, done
