


import lasagne.layers as L
import lasagne.nonlinearities as NL
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.core.network import MLP
from rllab.spaces import Box
from rllab.misc import ext
from rllab.policies.base import Policy
import theano
import theano.tensor as TT
import numpy as np
from sandbox.rocky.hrl.core.layers import GateLayer


class ReflectiveDeterministicMLPPolicy(Policy, LasagnePowered, Serializable):
    """
    This is intended to be used as a building block for the high-level policy.
    """

    def __init__(
            self,
            env_spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=None,
            action_network=None,
            gate_network=None,
            init_state=None,
            init_state_trainable=True,
            gated=True,
    ):
        raise DeprecationWarning("Deprecated. Use ReflectiveStochasticMLPPolicy!")
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        l_obs = L.InputLayer(
            shape=(None, obs_dim),
            name="obs"
        )

        l_prev_action = L.InputLayer(
            shape=(None, action_dim),
            name="prev_action"
        )

        # create network
        if action_network is None:
            action_network = MLP(
                input_layer=L.concat([l_obs, l_prev_action]),
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
            )
        if gated:
            if gate_network is None:
                gate_network = MLP(
                    input_layer=L.concat([l_obs, l_prev_action]),
                    output_dim=1,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=NL.sigmoid,
                )
        else:
            gate_network = None

        self.action_network = action_network
        self.gate_network = gate_network

        l_action = action_network.output_layer
        if gated:
            l_gate = gate_network.output_layer
            l_action = GateLayer(l_gate, l_action, l_prev_action)

        action_var = L.get_output(l_action)

        self.l_obs = l_obs
        self.l_prev_action = l_prev_action
        self.l_action = l_action

        self.f_action = ext.compile_function([l_obs.input_var, l_prev_action.input_var], action_var)

        if init_state is None:
            init_state = np.random.uniform(low=-1, high=1, size=(action_dim,))
        self.init_state_var = theano.shared(init_state, name="init_state")
        self.init_state_trainable = init_state_trainable
        self.prev_action = None

        super(ReflectiveDeterministicMLPPolicy, self).__init__(env_spec)

        output_layers = [l_action]

        LasagnePowered.__init__(self, [l_action])

    def reset(self):
        self.prev_action = self.init_state_var.get_value()

    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        prev_action = self.prev_action
        action = self.f_action([flat_obs], [prev_action])[0]
        self.prev_action = np.copy(action)
        return action, dict()

    @property
    def recurrent(self):
        return True

    def get_reparam_action_sym(self, obs_var, state_info_vars):
        # obs_var: N * T * S
        N = obs_var.shape[0]
        init_state = TT.tile(TT.reshape(self.init_state_var, (1, -1)), (N, 1))

        def step(cur_obs, prev_action):
            return L.get_output(self.l_action, {self.l_obs: cur_obs, self.l_prev_action: prev_action})

        permuted_actions, _ = theano.scan(step, sequences=obs_var.dimshuffle(1, 0, 2), outputs_info=init_state)
        return permuted_actions.dimshuffle(1, 0, 2)

    @property
    def distribution(self):
        return self

    @property
    def dist_info_keys(self):
        return []

    def get_params_internal(self, **tags):
        params = LasagnePowered.get_params_internal(self, **tags)
        if tags.get("trainable", self.init_state_trainable):
            params = params + [self.init_state_var]
        return params
