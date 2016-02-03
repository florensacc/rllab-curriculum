import numpy as np
from rllab.mdp.base import MDP, ControlMDP, SymbolicMDP
from rllab.core.serializable import Serializable
from rllab.mdp.proxy_mdp import ProxyMDP
from rllab.misc import autoargs
from rllab.misc.overrides import overrides

class NoisyObservationControlMDP(ProxyMDP, ControlMDP, Serializable):

    @autoargs.arg('obs_noise', type=float,
                  help='Noise added to the observations (note: this makes the '
                       'problem non-Markovian!)')
    def __init__(self,
                 mdp,
                 obs_noise=1e-1,
                 ):
        super(NoisyObservationControlMDP, self).__init__(mdp)
        ControlMDP.__init__(self)
        Serializable.quick_init(self, locals())
        self.obs_noise = obs_noise

    def get_obs_noise_scale_factor(self, obs):
        # return np.abs(obs)
        return np.ones_like(obs)

    def inject_obs_noise(self, obs):
        """
        Inject entry-wise noise to the observation. This should not change
        the dimension of the observation.
        """
        noise = self.get_obs_noise_scale_factor(obs) * self.obs_noise * \
                np.random.normal(size=obs.shape)
        return obs + noise

    def get_current_obs(self):
        return self.inject_obs_noise(self._mdp.get_current_obs())

    @overrides
    def reset(self):
        state, obs = self._mdp.reset()
        return state, self.inject_obs_noise(obs)

    @overrides
    def step(self, state, action):
        next_state, next_obs, reward, done = self._mdp.step(state, action)
        return next_state, self.inject_obs_noise(next_obs), reward, done

class DelayedActionControlMDP(ProxyMDP, ControlMDP, Serializable):

    @autoargs.arg('action_delay', type=int,
                  help='Time steps before action is realized')
    def __init__(self,
                 mdp,
                 action_delay=3,
                 ):
        assert action_delay > 0, "Should not use this mdp transformer"
        super(DelayedActionControlMDP, self).__init__(mdp)
        ControlMDP.__init__(self)
        Serializable.quick_init(self, locals())
        self.action_delay = action_delay
        self.original_state_len = mdp.state_shape[0]
        self.delayed_state = None

    @property
    @overrides
    def state_shape(self):
        return (self.original_state_len + self.action_delay*self.action_dim,)

    @overrides
    def reset(self):
        self._mdp.reset()
        self.delayed_state = np.zeros(self.state_shape)
        self.delayed_state[:self.original_state_len] = self._mdp.get_state()
        return self.get_state(), self.get_current_obs()

    @overrides
    def get_state(self):
        return self.delayed_state

    @overrides
    def step(self, state, action):
        original_state = state[:self.original_state_len]
        queued_action = state[self.original_state_len:][:self.action_dim]
        # import pdb; pdb.set_trace()
        next_original_state, next_obs, reward, done = self._mdp.step(original_state, queued_action)
        next_state = np.concatenate([
            next_original_state,
            state[self.original_state_len+self.action_dim:],
            action
        ])
        self.delayed_state = next_state
        return next_state, next_obs, reward, done

    @overrides
    def forward_dynamics(self, state, action, restore=True):
        raise NotImplementedError

