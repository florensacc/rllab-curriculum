import tensorflow as tf
import numpy as np
import scipy.stats

from rllab.misc.overrides import overrides

from sandbox.haoran.mddpg.core.neuralnet import NeuralNetwork
from sandbox.tuomas.utils.tf_util import mlp
from rllab.core.serializable import Serializable
from rllab.policies.base import Policy
from rllab.exploration_strategies.base import ExplorationStrategy
from sandbox.rocky.tf.core.parameterized import Parameterized
from sandbox.tuomas.mddpg.policies.nn_policy import NNPolicy


class StochasticNNPolicy(NNPolicy):
    """
    Note: we fix the input distribution to be standard normal
    """
    def __init__(self,
                 scope_name,
                 hidden_dims,
                 temperature_dim=0,
                 temperatures=None,
                 output_nonlinearity=tf.identity,
                 sample_dim=1,
                 freeze_samples=False,
                 K=1,
                 output_scale=1.0,
                 **kwargs):
        Serializable.quick_init(self, locals())

        self._sample_dim = sample_dim
        self._hidden_dims = hidden_dims
        self._output_scale = output_scale
        self._output_nonlinearity = output_nonlinearity

        self._freeze = freeze_samples
        if temperatures is not None:
            # TODO: temperature shouldn't be a property of a policy. Instead
            # it is a property of the algorithm.
            self._temp_dim = temperatures.shape[1]
            self._n_temps = temperatures.shape[0]
        else:
            self._temp_dim = 0

        self._temps = temperatures
        self._current_temp = 0

        # Freeze stuff
        self._K = K
        self._samples = np.random.randn(K, self._sample_dim)
        self.output_nonlinearity = output_nonlinearity
        self.output_scale = output_scale

        super(StochasticNNPolicy, self).__init__(scope_name, **kwargs)

        assert sample_dim == self.action_dim, """
            Unable to compute prob(action) if sample_dim != action_dim
        """

    def create_network(self, sample_input=None):
        self._create_pls(sample_input=sample_input)
        input_list = [self.observations_placeholder, self._sample_pl]
        if self._temp_dim > 0:
            input_list.append(self._temp_pl)

        all_inputs = tf.concat(concat_dim=1, values=input_list)

        self._pre_output = mlp(
            all_inputs,
            self.observation_dim + self._sample_dim + self._temp_dim,
            self._hidden_dims,
            output_layer_size=self.action_dim,
            nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.identity,
            W_initializer=None,
        )
        output = (
            self._output_scale * self.output_nonlinearity(self._pre_output)
        )

        # N: batch size, Da: action dim, Ds: sample dim
        self._Doutput_Dsample = tf.pack(
            [
                tf.gradients(output[:,i], self._sample_pl)[0]
                for i in range(self.action_dim)
            ],
            axis=1
        )  # N x Da x Ds
        Da = self._sample_dim
        self._sample_prob = np.power(2. * np.pi, -Da / 2.) * \
            tf.exp(-0.5 * tf.reduce_sum(
                tf.square(self._sample_pl),reduction_indices=1))
        self._action_prob = self._sample_prob / \
            tf.abs(tf.matrix_determinant(self._Doutput_Dsample))

        return output


    @property
    def pre_output(self):
        return self._pre_output

    def _create_pls(self, sample_input=None):
        if sample_input is None:
            self._sample_pl = tf.placeholder(
                tf.float32,
                shape=[None, self._sample_dim],
                name='actor_sample'
            )
        else:
            self._sample_pl = sample_input

        if self._temp_dim > 0:
            self._temp_pl = tf.placeholder(
                tf.float32,
                shape=[None, self._temp_dim],
                name='actor_temperature'
            )

    def get_feed_dict(self, observations, temperatures=None):
        N = observations.shape[0]
        feed = {self._sample_pl: self._get_input_samples(N),
                self.observations_placeholder: observations}
        if self._temp_dim > 0:
            if temperatures is None:
                temperatures = self._temps[self._current_temp][None]
                temperatures = np.tile(temperatures, (N, 1))

            assert (temperatures is not None
                    and temperatures.shape[1] == self._temp_dim)
            feed[self._temp_pl] = temperatures
        return feed

    def get_action(self, observation, temperature=None):
        observation = np.reshape(observation, (1, -1))
        if temperature is not None:
            temperature = np.reshape(temperature, (1, -1))
        return self.get_actions(observation, temperature)

    def get_actions(self, observations, temperatures=None, with_prob=False):
        feed = self.get_feed_dict(observations, temperatures)
        if not with_prob:
            actions = self.sess.run(self.output, feed)
            info = {}
        else:
            actions, action_probs = self.sess.run(
                [self.output, self._action_prob],
                feed,
            )
            info = {
                "prob": action_probs,
            }

        return actions, info

    def _get_input_samples(self, N):
        """
        Samples from input distribution q_0. Hardcoded as standard normal.

        :param N: Number of samples to be returned.
        :return: A numpy array holding N samples.
        """
        if self._freeze:
            indices = np.random.randint(low=0, high=self._K, size=N)
            samples = self._samples[indices]
            return samples
        else:
            return np.random.randn(N, self._sample_dim)

    def reset(self):
        if self._temp_dim > 0:
            self._current_temp = (self._current_temp + 1) % self._n_temps

    def compute_entropy(self, observations, n_sample):
        N = observations.shape[0]
        Do = self.observation_dim
        obs_copied = np.reshape(np.tile(observations, (1, n_sample)), (-1, Do))
        actions, info = self.get_actions(obs_copied, with_prob=True)
        action_probs = info["prob"].reshape(N, n_sample)
        TINY = 1e-8
        entropies = np.mean(-np.log(action_probs + TINY), axis=1)
        return entropies

class StochasticPolicyMaximizer(Parameterized, Serializable):
    """
        Same as StochasticNNPolicy, but samples several actions and picks
        the one that has maximal Q value.
    """

    def __init__(self, N, actor, critic):
        """

        :param N: Number of actions sampled.
        :param critic: Q function to be maximized.
        """
        Serializable.quick_init(self, locals())
        super(StochasticPolicyMaximizer, self).__init__()
        self._N = N
        self._critic = critic
        self._actor = actor
        self.sess = None  # This should be set elsewhere.

    @overrides
    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = np.tile(obs, (self._N, 1))

        actions, _ = self._actor.get_actions(obs)
        critic_feed = self._critic.get_feed_dict(obs, actions)
        q_vals = self.sess.run(self._critic.output, feed_dict=critic_feed)

        max_index = np.argmax(q_vals)

        return actions[max_index], {}

    @overrides
    def get_param_values(self):
        policy_params = self._actor.get_param_values()
        critic_params = self._critic.get_param_values()

        return (policy_params, critic_params)

    @overrides
    def set_param_values(self, params):

        self._actor.set_param_values(params[0])
        self._critic.set_param_values(params[1])

    @overrides
    def get_params_internal(self, **tags):
        actor_params = self._actor.get_params_internal()
        critic_params = self._critic.get_params_internal()
        return actor_params + critic_params

    @overrides
    def reset(self):
        pass

    @property
    def sess(self):
        if self._sess is None:
            self._sess = tf.get_default_session()
        return self._sess

    @sess.setter
    def sess(self, value):
        self._sess = value

class DummyExplorationStrategy(ExplorationStrategy):
    def get_action(self, t, observation, policy, **kwargs):
        action, _ = policy.get_action(observation)
        return action

    def reset(self):
        pass
