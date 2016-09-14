from __future__ import print_function
from __future__ import absolute_import
import numpy as np
from rocky.core.gym_ext import unflatten


class ParticleTrackingPolicy(object):
    def __init__(self, env):
        self.env = env
        self.model = env.world.model.model
        self.action_space = env.action_space
        self.agent_name = env.agent_name
        self.target_name = env.target_name

    def get_action(self, obs):
        agent_coords_offset = self.model.site_names.index(self.agent_name) * 3
        target_coords_offset = self.model.site_names.index(self.target_name) * 3
        agent_pos = obs[2][0, agent_coords_offset:agent_coords_offset + 2]
        target_pos = obs[2][0, target_coords_offset:target_coords_offset + 2]
        action = target_pos - agent_pos
        action -= np.squeeze(obs[1]) * 0.05
        return unflatten(self.action_space, action)

    def reset(self):
        pass
