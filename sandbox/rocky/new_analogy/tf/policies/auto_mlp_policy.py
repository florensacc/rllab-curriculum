from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.policies.base import StochasticPolicy
from rllab.misc import ext
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.regressors.auto_mlp_regressor import space_to_dist_dim, output_to_info, space_to_distribution
from sandbox.rocky.tf.spaces.discrete import Discrete
import tensorflow as tf


class AutoMLPPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
    ):
        """
        :param env_spec: A spec for the mdp.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param prob_network: manually specified network for this policy, other network params
        are ignored
        :return:
        """
        Serializable.quick_init(self, locals())

        # assert isinstance(env_spec.action_space, Discrete)

        obs_space = env_spec.observation_space
        obs_dim = obs_space.flat_dim
        action_space = env_spec.action_space

        with tf.variable_scope(name):
            network = MLP(
                name="policy",
                input_shape=(obs_dim,),
                output_dim=space_to_dist_dim(action_space),
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                # separate nonlinearities will be used for each component of the output
                output_nonlinearity=None,
            )

            self._l_out = network.output_layer
            self._l_obs = network.input_layer
            self._f_dist_info = tensor_utils.compile_function(
                [network.input_layer.input_var],
                output_to_info(L.get_output(network.output_layer), action_space)
            )

            self._dist = space_to_distribution(action_space)

            super(AutoMLPPolicy, self).__init__(env_spec)
            LayersPowered.__init__(self, [network.output_layer])

    @property
    def vectorized(self):
        return True

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars=None):
        return output_to_info(
            L.get_output(self._l_out, {self._l_obs: tf.cast(obs_var, tf.float32)}),
            self.action_space,
        )

    @overrides
    def dist_info(self, obs, state_infos=None):
        return self._f_dist_info(obs)

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        dist_info = self._f_dist_info(flat_obs)
        actions = self.distribution.sample(dist_info)
        return actions, dist_info

    @property
    def distribution(self):
        return self._dist
