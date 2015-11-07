from core.serializable import Serializable
import cgtcompat as theano
import cgtcompat.tensor as T

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

class SymbolicMDP(object):

    def step_symbolic(self, state, action):
        raise NotImplementedError

    def step(self, state, action):
        if not hasattr(self, '_f_step'):
            s = T.vector('s')
            a = T.vector('a')
            self._f_step = theano.function([s, a], self.step_symbolic(s, a), allow_input_downcast=True, on_unused_input='ignore')
        return tuple(self._f_step(state, action))

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
