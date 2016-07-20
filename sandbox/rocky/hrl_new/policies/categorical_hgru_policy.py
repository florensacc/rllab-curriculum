import numpy as np
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.network import GRUNetwork, MLP
from sandbox.rocky.tf.distributions.recurrent_categorical import RecurrentCategorical
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.policies.base import StochasticPolicy

from rllab.core.serializable import Serializable
from rllab.misc import special
from rllab.misc.overrides import overrides


class IdentityNetwork(object):
    def __init__(self, layer):
        self.layer = layer

    @property
    def output_layer(self):
        return self.layer


class MiniGRUNetwork(object):
    def __init__(self, name, input_shape, hidden_dim, hidden_nonlinearity=tf.nn.relu,
                 input_var=None, input_layer=None):
        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(None, None) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            l_step_input = L.InputLayer(shape=(None,) + input_shape, name="step_input")
            l_step_prev_hidden = L.InputLayer(shape=(None, hidden_dim), name="step_prev_hidden")

            l_gru = L.GRULayer(l_in, num_units=hidden_dim, hidden_nonlinearity=hidden_nonlinearity, name="gru")

            l_step_hidden = l_gru.get_step_layer(l_step_input, l_step_prev_hidden, name="step_hidden")

            self._l_in = l_in
            self._hid_init_param = l_gru.h0
            self._l_gru = l_gru
            self._l_step_input = l_step_input
            self._l_step_prev_hidden = l_step_prev_hidden
            self._l_step_hidden = l_step_hidden

    @property
    def input_layer(self):
        return self._l_in

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def hidden_layer(self):
        return self._l_gru

    @property
    def step_input_layer(self):
        return self._l_step_input

    @property
    def step_prev_hidden_layer(self):
        return self._l_step_prev_hidden

    @property
    def step_hidden_layer(self):
        return self._l_step_hidden

    @property
    def hid_init_param(self):
        return self._hid_init_param


class CategoricalHGRUPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_dim=32,
            feature_hidden_sizes=(32,),
            feature_dim=32,
            skip_hidden_sizes=(32,),
            skip_feature_dim=32,
            hidden_nonlinearity=tf.tanh):
        """
        :param env_spec: A spec for the env.
        :param hidden_dim: dimension of hidden layer
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        with tf.variable_scope(name):
            assert isinstance(env_spec.action_space, Discrete)
            Serializable.quick_init(self, locals())
            super(CategoricalHGRUPolicy, self).__init__(env_spec)

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            l_obs = L.InputLayer(
                shape=(None, None, obs_dim),
                name="obs"
            )
            l_prev_action = L.InputLayer(
                shape=(None, None, action_dim),
                name="action"
            )

            l_input = L.concat([l_obs, l_prev_action], name="input", axis=2)
            input_dim = obs_dim + action_dim

            l_flat_obs = L.reshape(l_obs, (-1, obs_dim), name="flat_obs")
            l_flat_prev_action = L.reshape(l_prev_action, (-1, action_dim), name="flat_prev_action")

            l_flat_input = L.concat([l_flat_obs, l_flat_prev_action], name="flat_input", axis=1)

            feature_network = MLP(
                input_shape=(input_dim,),
                input_layer=l_flat_input,
                hidden_nonlinearity=hidden_nonlinearity,
                hidden_sizes=feature_hidden_sizes,
                output_dim=feature_dim,
                output_nonlinearity=hidden_nonlinearity,
                name="feature_network"
            )

            l_flat_feature = feature_network.output_layer
            l_feature = L.OpLayer(
                l_flat_feature,
                extras=[l_input],
                name="reshape_feature",
                op=lambda flat_feature, input: tf.reshape(
                    flat_feature,
                    tf.pack([tf.shape(input)[0], tf.shape(input)[1], feature_dim])
                ),
                shape_op=lambda _, input_shape: (input_shape[0], input_shape[1], feature_dim)
            )

            hidden_network = MiniGRUNetwork(
                input_shape=(feature_dim,),
                input_layer=l_feature,
                hidden_dim=hidden_dim,
                hidden_nonlinearity=hidden_nonlinearity,
                name="prob_network"
            )

            skip_network = MLP(
                input_shape=(obs_dim,),
                input_layer=l_flat_obs,
                hidden_nonlinearity=hidden_nonlinearity,
                hidden_sizes=skip_hidden_sizes,
                output_dim=skip_feature_dim,
                output_nonlinearity=hidden_nonlinearity,
                name="skip_network",
            )
            l_flat_skip_feature = skip_network.output_layer

            l_flat_hidden = L.reshape(hidden_network.hidden_layer, (-1, hidden_dim), name="flat_hidden")

            l_concat = L.concat([l_flat_hidden, l_flat_skip_feature], axis=1, name="concat")

            l_flat_prob = L.DenseLayer(
                l_concat,
                num_units=action_dim,
                name="flat_prob",
                nonlinearity=tf.nn.softmax,
            )

            l_prob = L.OpLayer(
                l_flat_prob,
                extras=[l_input],
                name="reshape_prob",
                op=lambda flat_prob, input: tf.reshape(
                    flat_prob,
                    tf.pack([tf.shape(input)[0], tf.shape(input)[1], action_dim])
                ),
                shape_op=lambda _, input_shape: (input_shape[0], input_shape[1], action_dim)
            )

            self.hidden_network = hidden_network
            self.feature_network = feature_network
            self.skip_network = skip_network
            self.l_input = l_input
            self.l_obs = l_obs
            self.l_prev_action = l_prev_action
            self.l_flat_prob = l_flat_prob
            self.l_prob = l_prob

            flat_obs_var = tf.placeholder(dtype=tf.float32, shape=(None, obs_dim), name="flat_obs")
            flat_prev_action_var = tf.placeholder(dtype=tf.float32, shape=(None, action_dim), name="flat_prev_action")
            flat_feature_var = L.get_output(
                l_flat_feature, {
                    l_flat_obs: flat_obs_var,
                    l_flat_prev_action: flat_prev_action_var
                }
            )

            flat_hidden_var = L.get_output(hidden_network.step_hidden_layer, {
                l_flat_obs: flat_obs_var,
                l_flat_prev_action: flat_prev_action_var,
                hidden_network.step_input_layer: flat_feature_var
            })

            self.f_step_prob = tensor_utils.compile_function(
                [
                    flat_obs_var,
                    flat_prev_action_var,
                    hidden_network.step_prev_hidden_layer.input_var
                ],
                [
                    L.get_output(
                        l_flat_prob,
                        {
                            l_flat_hidden: flat_hidden_var,
                            l_flat_obs: flat_obs_var,
                            l_flat_prev_action: flat_prev_action_var,
                        }
                    ),
                    flat_hidden_var,
                ],

                #     hidden_network.step_hidden_layer
                # ], {l_input: tf.expand_dims(flat_input_var, 1),
                #     l_flat_input: flat_input_var,
                #     hidden_network.step_input_layer: flat_feature_var})
            )

            self.input_dim = input_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim

            self.prev_actions = None
            self.prev_hiddens = None
            self.dist = RecurrentCategorical(env_spec.action_space.n)

            LayersPowered.__init__(self, [l_prob])

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars):
        n_batches = tf.shape(obs_var)[0]
        n_steps = tf.shape(obs_var)[1]
        obs_var = tf.reshape(obs_var, tf.pack([n_batches, n_steps, -1]))
        obs_var = tf.cast(obs_var, tf.float32)
        return dict(
            prob=L.get_output(self.l_prob, {
                self.l_obs: obs_var,
                self.l_prev_action: tf.cast(state_info_vars["prev_action"], tf.float32)
            })
        )

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self.prev_actions is None or len(dones) != len(self.prev_actions):
            self.prev_actions = np.zeros((len(dones), self.action_space.flat_dim))
            self.prev_hiddens = np.zeros((len(dones), self.hidden_dim))

        self.prev_actions[dones] = 0.
        self.prev_hiddens[dones] = self.hidden_network.hid_init_param.eval()  # get_value()

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.iteritems()}

    @overrides
    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        probs, hidden_vec = self.f_step_prob(flat_obs, self.prev_actions, self.prev_hiddens)
        actions = special.weighted_sample_n(probs, np.arange(self.action_space.n))
        prev_actions = self.prev_actions
        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_hiddens = hidden_vec
        agent_info = dict(prob=probs, prev_action=np.copy(prev_actions))
        return actions, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self.dist

    @property
    def state_info_specs(self):
        return [
            ("prev_action", (self.action_dim,)),
        ]

    # def reg_sym(self, obs_var, action_var, valid_var, dist_info_vars, state_info_vars, **kwargs):
    #
    #     logli = self.distribution.log_likelihood_sym(action_var, dist_info_vars)
    #     mean_logli = tf.reduce_sum(logli * valid_var) / tf.reduce_sum(valid_var)
    #     saliency = tf.reduce_mean(tf.square(tf.gradients(mean_logli, state_info_vars['prev_action'])[0]))
    #     return saliency
        # import ipdb; ipdb.set_trace()
        # self.dist_info_sym(obs_var, state_info_vars)

        # regularize the change in the hidden units
        # return 0
        # pass

    def log_diagnostics(self, samples_data):
        if not hasattr(self, 'f_debug'):
            # return
            action_var = self.action_space.new_tensor_variable('action', extra_dims=2)
            obs_var = self.observation_space.new_tensor_variable('obs', extra_dims=2)
            dist_info_vars_list = self.observation_space.new_tensor_variable('obs', extra_dims=2)
            valid_var = tf.placeholder(tf.float32, shape=(None, None), name='valid')
            # dist_info_vars = {
            #     k: tf.placeholder(tf.float32, shape=[None, None] + list(shape), name=k)
            #     for k, shape in self.distribution.dist_info_specs
            #     }
            # dist_info_vars_list = [dist_info_vars[k] for k in self.distribution.dist_info_keys]

            state_info_vars = {
                k: tf.placeholder(tf.float32, shape=[None, None] + list(shape), name=k)
                for k, shape in self.state_info_specs
                }
            state_info_vars_list = [state_info_vars[k] for k in self.state_info_keys]
            dist_info_vars = self.dist_info_sym(obs_var, state_info_vars)

            logli = self.distribution.log_likelihood_sym(action_var, dist_info_vars)
            mean_logli = tf.reduce_sum(logli * valid_var) / tf.reduce_sum(valid_var)
            saliency = tf.reduce_mean(tf.square(tf.gradients(mean_logli, state_info_vars['prev_action'])[0]))

            self.f_debug = tensor_utils.compile_function(
                [obs_var, action_var, valid_var] + state_info_vars_list,
                saliency
            )

        agent_infos = samples_data['agent_infos']
        state_info_list = [agent_infos[k] for k in self.state_info_keys]

        saliency = self.f_debug(
            samples_data['observations'], samples_data['actions'], samples_data['valids'], *state_info_list)
        from rllab.misc import logger
        logger.record_tabular('Saliency', saliency)

        # W = self.l_flat_prob.W.eval()
        # W_hidden = W[:self.hidden_dim]
        # W_skipped = W[self.hidden_dim:]
        # logger.record_tabular('W_hidden.norm', np.linalg.norm(W_hidden))
        # logger.record_tabular('W_skipped.norm', np.linalg.norm(W_skipped))
