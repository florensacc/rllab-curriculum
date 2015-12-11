from rllab.mjcapi.john_mjc.mjc2 import MJCMDP
from rllab.mdp.base import ControlMDP
from rllab.misc.overrides import overrides
from rllab.misc.serializable import Serializable
from rllab.mjcapi.john_mjc.config import floatX
import numpy as np


class WrapperMDP(ControlMDP, Serializable):

    def __init__(self):
        self._mdp = MJCMDP(self.BASE_NAME)
        self._state = None
        self._action = None
        Serializable.__init__(self)

    @overrides
    def step(self, state, action):
        state = np.array(state).astype(floatX).reshape(-1)
        lb, ub = self.action_bounds
        action = np.array(action).astype(floatX).reshape(-1)
        # scale action
        action = action * (ub - lb) + lb
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
    def action_dim(self):
        return self._mdp.ctrl_dim()

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
            states = [np.reshape(state, (-1)) for state in states]
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
PlanarHumanoidMDP = type('PlanaerHumanoidMDP', (WrapperMDP,), dict(BASE_NAME='planar_humanoid'))
TarsMDP = type('TarsMDP', (WrapperMDP,), dict(BASE_NAME='tars'))
AntMDP = type('AntMDP', (WrapperMDP,), dict(BASE_NAME='ant'))
AtlasMDP = type('AtlasMDP', (WrapperMDP,), dict(BASE_NAME='atlas'))
IcmlHumanoidMDP = type('IcmlHumanoidMDP', (WrapperMDP,), dict(BASE_NAME='icml_humanoid'))
IcmlHumanoidTrackingMDP = type('IcmlHumanoidTrackingMDP', (WrapperMDP,), dict(BASE_NAME='icml_humanoid_tracking'))
Humanoid3DAmputatedMDP = type('Humanoid3DAmputatedMDP', (WrapperMDP,), dict(BASE_NAME='3d_humanoid_amputated')) # Broken: dim mismatch
IcmlHumanoidJumperMDP = type('IcmlHumanoidJumperMDP', (WrapperMDP,), dict(BASE_NAME='icml_humanoid_jumper'))
Humanoid3DMDP = type('Humanoid3DMDP', (WrapperMDP,), dict(BASE_NAME='3d_humanoid'))
Humanoid3DSitMDP = type('Humanoid3DSitMDP', (WrapperMDP,), dict(BASE_NAME='3d_humanoid_sit'))
Humanoid3DStandMDP = type('Humanoid3DStandMDP', (WrapperMDP,), dict(BASE_NAME='3d_humanoid_stand'))
Humanoid3DStandAndWalkMDP = type('Humanoid3DStandAndWalkMDP', (WrapperMDP,), dict(BASE_NAME='3d_humanoid_stand_and_walk'))
HumanWalkingMDP = type('HumanWalkingMDP', (WrapperMDP,), dict(BASE_NAME='human_walking')) # Broken: missing xml file
SwimmerMDP = type('SwimmerMDP', (WrapperMDP,), dict(BASE_NAME='swimmer'))
HopperMDP = type('HopperMDP', (WrapperMDP,), dict(BASE_NAME='hopper'))
Walker2DMDP = type('Walker2DMDP', (WrapperMDP,), dict(BASE_NAME='walker2d'))
MuscleWalker2DMDP = type('MuscleWalker2DMDP', (WrapperMDP,), dict(BASE_NAME='musclewalker2d')) # Broken: muscle unrecognized
IgorWalker2DMDP = type('IgorWalker2DMDP', (WrapperMDP,), dict(BASE_NAME='igorwalker2d'))
BallHopperMDP = type('BallHopperMDP', (WrapperMDP,), dict(BASE_NAME='ball_hopper'))
