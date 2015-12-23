import theano
import theano.tensor as TT
from rllab.misc.ext import cached_function, lazydict
from rllab.misc import autoargs


class MDP(object):

    def step(self, state, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def action_dim(self):
        raise NotImplementedError

    @property
    def observation_shape(self):
        raise NotImplementedError

    @property
    def action_dtype(self):
        raise NotImplementedError

    @property
    def observation_dtype(self):
        raise NotImplementedError

    def start_viewer(self):
        pass

    def stop_viewer(self):
        pass

    def plot(self, states=None, actions=None, pause=False):
        raise NotImplementedError

    @classmethod
    @autoargs.add_args
    def add_args(cls, parser):
        pass

    @classmethod
    @autoargs.new_from_args
    def new_from_args(cls, args):
        pass

    def log_extra(self, logger, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    def action_from_keys(self, keys):
        raise NotImplementedError


class ControlMDP(MDP):

    def cost(self, state, action):
        raise NotImplementedError

    def final_cost(self, state):
        raise NotImplementedError

    def forward_dynamics(self, state, action, restore=True):
        raise NotImplementedError

    @property
    def state_shape(self):
        raise NotImplementedError

    @property
    def state_bounds(self):
        raise NotImplementedError

    @property
    def action_bounds(self):
        raise NotImplementedError


class SymbolicMDP(ControlMDP):

    def __init__(self):
        super(SymbolicMDP, self).__init__()
        self._state_sym = TT.vector('state')
        self._action_sym = TT.vector('action')

        # placeholder for cached compiled functions
        self._f_obs = None
        self._f_forward = None
        self._f_step = None
        self._f_reset = None
        self._f_cost = None
        self._f_final_cost = None
        self._grad_hints = None

        self._state = self.reset()
        self._action = None

    @property
    def state(self):
        return self._state

    @property
    def grad_hints(self):
        if self._grad_hints is None:
            s = self._state_sym
            a = self._action_sym
            lookup = lazydict(
                f=lambda: self.forward_sym(s, a),
                c=lambda: self.cost_sym(s, a),
                cf=lambda: self.final_cost_sym(s)
            )
            # we need first order approximation for the dynamics,
            # and second order approximation for the cost
            self._grad_hints = lazydict(
                # gradients for forward dynamics
                df_dx=lambda: cached_function([s, a], theano.gradient.jacobian(lookup['f'], s)),
                df_du=lambda: cached_function([s, a], theano.gradient.jacobian(lookup['f'], a)),
                # gradients for cost
                dc_dx=lambda: cached_function([s, a], theano.gradient.grad(lookup['c'], s)),
                dc_du=lambda: cached_function([s, a], theano.gradient.grad(lookup['c'], a)),
                dc_dxx=lambda: cached_function([s, a], theano.gradient.jacobian(theano.gradient.grad(lookup['c'], s), s)),
                dc_duu=lambda: cached_function([s, a], theano.gradient.jacobian(theano.gradient.grad(lookup['c'], a), a)),
                dc_dxu=lambda: cached_function([s, a], theano.gradient.jacobian(theano.gradient.grad(lookup['c'], s), a)),
                # gradients for final cost
                dcf_dx=lambda: cached_function([s], theano.gradient.grad(lookup['cf'], s)),
                dcf_dxx=lambda: cached_function([s], theano.gradient.jacobian(theano.gradient.grad(lookup['cf'], s), s)),
            )
        return self._grad_hints

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
        if self._f_reset is None:
            self._f_reset = cached_function([], self.reset_sym())
        return self._f_reset()

    def step(self, state, action):
        if self._f_step is None:
            s = self._state_sym
            a = self._action_sym
            self._f_step = cached_function([s, a], self.step_sym(s, a))
        ns, obs, reward, done = self._f_step(state, action)
        self._state = ns
        self._action = action
        return ns, obs, reward, done

    def get_observation(self, state):
        if self._f_obs is None:
            s = self._state_sym
            a = self._action_sym
            self._f_obs = cached_function([s, a], self.observation_sym(s))
        return self._f_obs(state)

    def forward(self, state, action):
        if self._f_forward is None:
            s = self._state_sym
            a = self._action_sym
            self._f_forward = cached_function([s, a], self.forward_sym(s, a))
        return self._f_forward(state, action)

    def cost(self, state, action):
        if self._f_cost is None:
            s = self._state_sym
            a = self._action_sym
            self._f_cost = cached_function([s, a], self.cost_sym(s, a))
        return self._f_cost(state, action)

    def final_cost(self, state):
        if self._f_final_cost is None:
            s = self._state_sym
            self._f_final_cost = cached_function([s], self.final_cost_sym(s))
        return self._f_final_cost(state)
