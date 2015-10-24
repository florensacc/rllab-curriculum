from core.serializable import Serializable

class MDP(object):

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def n_actions(self):
        raise NotImplementedError

    @property
    def observation_shape(self):
        raise NotImplementedError

    @property
    def action_dtype(self):
        raise NotImplementedError


class ControlMDP(MDP):
    
    def __init__(self, horizon):
        self.horizon = horizon
        super(MDP, self).__init__()
    
    def cost(self, state, action):
        raise NotImplementedError

    def final_cost(self, state):
        raise NotImplementedError

    def forward_dynamics(self, state, action):
        raise NotImplementedError
