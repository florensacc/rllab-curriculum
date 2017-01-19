import tensorflow as tf
import numpy as np

from sandbox.haoran.mddpg.core.neuralnet import NeuralNetwork
from sandbox.tuomas.utils.tf_util import mlp
from rllab.core.serializable import Serializable
from rllab.policies.base import Policy
from rllab.exploration_strategies.base import ExplorationStrategy


class StochasticNNPolicy(NeuralNetwork, Policy):
    def __init__(self,
                 scope_name,
                 observation_dim,
                 action_dim,
                 hidden_dims,
                 W_initializer=None,
                 output_nonlinearity=tf.identity,
                 sample_dim=1,
                 freeze_samples=False,
                 K=1,
                 output_scale=1.0,
                 **kwargs):
        Serializable.quick_init(self, locals())

        self._obs_dim = observation_dim
        self._action_dim = action_dim
        self._sample_dim = sample_dim
        self._rnd = np.random.RandomState()
        self._freeze = freeze_samples

        with tf.variable_scope(scope_name) as variable_scope:
            super(StochasticNNPolicy, self).__init__(
                variable_scope.original_name_scope, **kwargs
            )
            self._create_pls()
            all_inputs = tf.concat(concat_dim=1,
                                   values=(self._obs_pl, self._sample_pl))

            self._pre_output = mlp(
                all_inputs,
                observation_dim + self._sample_dim,
                hidden_dims,
                output_layer_size=action_dim,
                nonlinearity=tf.nn.relu,
                output_nonlinearity=tf.identity,
                W_initializer=W_initializer
            )
            self._output = output_scale * output_nonlinearity(self._pre_output)
            self.variable_scope = variable_scope

        # Freeze stuff
        self._K = K
        self._samples = np.random.randn(K, self._sample_dim)
        self._k = np.random.randint(0, self._K)
        self.output_nonlinearity = output_nonlinearity
        self.output_scale = output_scale

    @property
    def pre_output(self):
        return self._pre_output

    def _create_pls(self):
            self._obs_pl = tf.placeholder(
                tf.float32,
                shape=[None, self._obs_dim],
                name='actor_obs'
            )
            self._sample_pl = tf.placeholder(
                tf.float32,
                shape=[None, self._sample_dim],
                name='actor_sample'
            )

            # Give another name for backward compatibility
            # TODO: should not access these directly.
            self.observations_placeholder = self._obs_pl

            self._input = self._obs_pl

    def get_feed_dict(self, observations):
        if type(observations) == list:
            observations = np.array(observations)
        N = observations.shape[0]
        feed = {self._sample_pl: self._get_input_samples(N),
                self._obs_pl: observations}
        return feed

    def get_action(self, observation):
        return self.get_actions([observation])

    def get_actions(self, observations):
        feed = self.get_feed_dict(observations)
        #if type(observations) is not list and observations.shape[0] == 16:
        #    import pdb; pdb.set_trace()
        return self.sess.run(self.output, feed), {}

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
            #import pdb; pdb.set_trace()
            return self._rnd.randn(N, self._sample_dim)
            #self._rnd.seed(0)

        #return self._rnd.randn(N, self._sample_dim)

    def reset(self):
        self._k = np.random.randint(0, self._K)


class DummyExplorationStrategy(ExplorationStrategy):
    def get_action(self, t, observation, policy, **kwargs):
        action, _ = policy.get_action(observation)
        return action

    def reset(self):
        pass
