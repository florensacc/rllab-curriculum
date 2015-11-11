from core.serializable import Serializable
import cgtcompat as theano
import cgtcompat.tensor as T
import cPickle as pickle
from path import Path
import sys
from misc.ext import cached_function

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

    @property
    def state_shape(self):
        raise NotImplementedError

class SymbolicMDP(ControlMDP):

    def __init__(self, horizon):
        super(SymbolicMDP, self).__init__(horizon)
        s = T.vector('state', fixed_shape=self.state_shape)
        a = T.vector('action', fixed_shape=(self.n_actions,))
        ns = T.vector('next_state', fixed_shape=self.state_shape)
        self._f_obs = cached_function([s, a], self.observation_sym(s))
        self._f_forward = cached_function([s, a], self.forward_sym(s, a))
        self._f_reward = cached_function([s, a], self.reward_sym(s, a))
        self._f_done = cached_function([ns], self.done_sym(ns))
        self._f_step = cached_function([s, a], self.step_sym(s, a))
        self._f_reset = cached_function([], self.reset_sym())
        self._state = self.reset()
        self._action = None

    @property
    def state(self):
        return self._state

    @property
    def action(self):
        return self._action

    def observation_sym(self, state):
        raise NotImplementedError

    def forward_sym(self, state, action):
        raise NotImplementedError

    def reward_sym(self, state, action):
        raise NotImplementedError

    def cost_sym(self, state, action):
        raise NotImplementedError

    def final_cost_sym(self, state):
        raise NotImplementedError

    def done_sym(self, next_state):
        raise NotImplementedError

    def step_sym(self, state, action):
        ns = self.forward_sym(state, action)
        obs = self.observation_sym(ns)
        reward = self.reward_sym(state, action)
        done = self.done_sym(ns)
        return ns, obs, reward, done

    def reset(self):
        res = self._f_reset()
        return res

    def step(self, state, action):
        ns, obs, reward, done = self._f_step(state, action)
        self._state = ns
        self._action = action
        return ns, obs, reward, done

    def get_observation(self, state):
        return self._f_obs(state)

    def forward(self, state, action):
        return self._f_forward(state, action)
