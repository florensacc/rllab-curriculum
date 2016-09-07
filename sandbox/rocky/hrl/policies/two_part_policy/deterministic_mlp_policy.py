


import lasagne.layers as L
import lasagne.nonlinearities as NL
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.core.network import MLP
from rllab.spaces import Box
from rllab.misc import ext
from rllab.policies.base import Policy


class DeterministicMLPPolicy(Policy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=None,
            action_network=None):
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        # create network
        if action_network is None:
            action_network = MLP(
                input_shape=(obs_dim,),
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
                name="action_network"
            )
        self.action_network = action_network

        l_obs = action_network.input_layer
        l_output = action_network.output_layer
        action_var = L.get_output(l_output)

        self.l_obs = l_obs
        self.l_output = l_output
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity

        self.f_action = ext.compile_function([l_obs.input_var], action_var)

        super(DeterministicMLPPolicy, self).__init__(env_spec)
        LasagnePowered.__init__(self, [l_output])

    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        action = self.f_action([flat_obs])[0]
        return action, dict()

    # def get_actions(self, observations):
    #     flat_obs = self.observation_space.flatten_n(observations)
    #     return self.f_action(flat_obs), dict()

    def get_reparam_action_sym(self, obs_var, state_info_vars):
        return L.get_output(self.l_output, {self.l_obs: obs_var})

    @property
    def distribution(self):
        return self

    @property
    def dist_info_keys(self):
        return []
