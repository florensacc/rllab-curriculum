from rllab.mjcapi.john_mjc.mjc import get_mjc_mdp_class
from rllab.mdp.base import ControlMDP
from rllab.misc.overrides import overrides
from rllab.mjcapi.john_mjc.config import floatX
from rllab.core.serializable import Serializable
import numpy as np


class WrapperMDP(ControlMDP, Serializable):

    def __init__(self):
        self._mdp = get_mjc_mdp_class(self.BASE_NAME)
        self._state = None
        self._action = None
        Serializable.__init__(self)

    @overrides
    def step(self, state, action):
        state = np.array(state).astype(floatX).reshape((1, -1))
        lb, ub = self.action_bounds
        action = np.array(action).astype(floatX).reshape((1, -1))
        result = self._mdp.call({'x': state, 'u': action})
        next_state = result["x"].reshape(-1)
        next_obs = result["o"].reshape(-1)
        done = bool(result["done"])
        reward = -result["c"].sum()
        self._state = next_state
        return next_state, next_obs, reward, done

    @property
    @overrides
    def state_shape(self):
        return (self._mdp.state_dim(),)

    @property
    @overrides
    def observation_shape(self):
        return (self._mdp.obs_dim(),)

    @property
    @overrides
    def observation_dtype(self):
        return 'float32'

    @property
    @overrides
    def action_dim(self):
        return self._mdp.ctrl_dim()

    @property
    @overrides
    def action_dtype(self):
        return 'float32'

    @overrides
    def reset(self):
        data = self._mdp.initialize_mdp_arrays()
        state = data["x"].reshape(-1)
        obs = data["o"].reshape(-1)
        self._state = state
        return state, obs

    def plot(self, states=None, actions=None):
        if states is None:
            if self._state is not None:
                states = [self._state]
            else:
                states = []
        if len(states) > 0:
            states = [np.reshape(state, (1, -1)) for state in states]
            self._mdp.plot({"x": np.vstack(states)})

    @property
    @overrides
    def state_bounds(self):
        return self._mdp.state_bounds()

    @property
    @overrides
    def action_bounds(self):
        bounds = self._mdp.ctrl_bounds().reshape(-1)
        lb = bounds[:self.action_dim]
        ub = bounds[self.action_dim:]
        return lb, ub


# Shortcut for declaring subclasses
SwimmerMDP = type('SwimmerMDP', (WrapperMDP,), dict(BASE_NAME='3swimmer'))
Hopper4BallMDP = type('Hopper4BallMDP', (WrapperMDP,), dict(BASE_NAME='hopper4ball'))
Walker2DMDP = type('Walker2DMDP', (WrapperMDP,), dict(BASE_NAME='walker2d'))
TripodMDP = type('TripodMDP', (WrapperMDP,), dict(BASE_NAME='tripod'))
Human3DMDP = type('Human3DMDP', (WrapperMDP,), dict(BASE_NAME='human3d'))
BvhModelMDP = type('BvhModelMDP', (WrapperMDP,), dict(BASE_NAME='bvhmodel'))
