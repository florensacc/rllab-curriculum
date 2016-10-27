import numpy as np
from tensorflow.python.ops.rnn_cell import RNNCell

import sandbox.rocky.tf.core.layers as L
import sandbox.rocky.analogy.core.layers as LL
import tensorflow as tf

from sandbox.rocky.analogy.rnn_cells import GRUCell, linear
from sandbox.rocky.neural_learner.episodic.episodic_cell import EpisodicCell
from sandbox.rocky.neural_learner.episodic.episodic_network_builder import EpisodicNetworkBuilder
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.distributions.recurrent_categorical import RecurrentCategorical
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.spaces import Product, Box
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.policies.base import StochasticPolicy

from rllab.core.serializable import Serializable
from rllab.misc import special
from rllab.misc.overrides import overrides


class EpisodicRNNPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            cell=None,
            network_builder=None,
    ):
        Serializable.quick_init(self, locals())

        if network_builder is None:
            network_builder = EpisodicNetworkBuilder(env_spec)

        if cell is None:
            cell = EpisodicCell(GRUCell(num_units=128, activation=tf.nn.relu, weight_normalization=True))

        with tf.variable_scope(name):
            assert isinstance(env_spec.action_space, Discrete)
            StochasticPolicy.__init__(self, env_spec)

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            l_obs_input = L.InputLayer(
                shape=(None, None, self.observation_space.flat_dim),
                name="obs"
            )

            l_obs_components = network_builder.split_obs_layer(l_obs_input)

            l_raw_obs, l_prev_action, l_reward, l_terminal = l_obs_components
            l_obs_feature = network_builder.new_obs_feature_layer(l_raw_obs)
            l_action_feature = network_builder.new_action_feature_layer(l_prev_action)

            self.obs_feature_dim = l_obs_feature.output_shape[-1]
            self.action_feature_dim = l_action_feature.output_shape[-1]

            l_feature = L.concat([l_obs_feature, l_action_feature, l_reward, l_terminal], axis=2)

            l_rnn = network_builder.new_rnn_layer(
                l_feature,
                cell=cell,
            )
            l_prob = L.TemporalUnflattenLayer(
                L.DenseLayer(
                    L.TemporalFlattenLayer(l_rnn),
                    num_units=action_dim,
                    nonlinearity=tf.nn.softmax,
                ),
                ref_layer=l_obs_input
            )

            self.l_obs_input = l_obs_input
            self.l_obs_components = l_obs_components
            self.l_feature = l_feature
            self.l_rnn = l_rnn
            self.l_prob = l_prob

            self.action_dim = action_dim

            self.state_dim = l_rnn.state_dim

            self.prev_states = None
            self.dist = RecurrentCategorical(env_spec.action_space.n)

            flat_obs_var = env_spec.observation_space.new_tensor_variable("flat_obs", extra_dims=1)
            obs_var = tf.expand_dims(flat_obs_var, 1)

            prev_state_var = tf.placeholder(
                dtype=tf.float32,
                shape=(None, l_rnn.cell.state_size),
                name="prev_state"
            )
            recurrent_state_output = dict()

            prob_var = self.dist_info_sym(
                obs_var,
                recurrent_state={l_rnn: prev_state_var},
                recurrent_state_output=recurrent_state_output,
            )["prob"][:, 0, :]

            self.f_step = tensor_utils.compile_function(
                inputs=[flat_obs_var, prev_state_var],
                outputs=[prob_var, recurrent_state_output[l_rnn]],
            )

            LayersPowered.__init__(self, [l_prob])

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars=None, **kwargs):
        prob_var = L.get_output(
            self.l_prob,
            {self.l_obs_input: obs_var},
            **kwargs
        )
        return dict(prob=prob_var)

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self.prev_states is None or len(dones) != len(self.prev_states):
            self.prev_states = np.zeros((len(dones), self.state_dim))

        if np.any(dones):
            self.prev_states[dones] = 0.

    @overrides
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    @overrides
    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        probs, state_vec = self.f_step(flat_obs, self.prev_states)
        actions = special.weighted_sample_n(probs, np.arange(self.action_space.n))
        self.prev_states = state_vec
        agent_info = dict(prob=probs)
        return actions, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self.dist


def _main():
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv
    from sandbox.rocky.neural_learner.envs.random_tabular_mdp_env import RandomTabularMDPEnv
    env = TfEnv(
        MultiEnv(
            wrapped_env=RandomTabularMDPEnv(
                n_states=10,
                n_actions=5,
            ),
            n_episodes=10,
            episode_horizon=10,
            discount=0.99,
        )
    )

    policy = EpisodicRNNPolicy(name="policy", env_spec=env.spec)


if __name__ == "__main__":
    _main()
