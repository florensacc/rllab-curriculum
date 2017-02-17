import theano
import theano.tensor as TT
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np
import joblib
import tensorflow as tf

from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.spaces import Box

from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import ext
from rllab.distributions.diagonal_gaussian import DiagonalGaussian

from sandbox.tuomas.mddpg.policies.stochastic_policy \
    import StochasticNNPolicy as TfStochasticNNPolicy


class StochasticNNPolicy(StochasticPolicy, LasagnePowered, Serializable):
    """
    A policy that takes the observation and noise as inputs and outputs an
    action. By default the input noise is fixed to standard Gaussian.
    Notice that the output distribution is intractable.
    """
    def __init__(
            self,
            env_spec,
            hidden_dims,
            hidden_nonlinearity=NL.rectify,
            output_nonlinearity=NL.tanh,
            sample_dim=None,
            freeze_samples=False,
            K=1,
    ):
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        if sample_dim is None:
            self._sample_dim = self._action_dim
        else:
            self._sample_dim = sample_dim
        self._hidden_dims = hidden_dims
        self._output_nonlinearity = output_nonlinearity
        self._freeze_samples = freeze_samples
        self._K = K
        self._samples = np.random.randn(K, self._sample_dim)


        # Usually we enforce the constraint below so that the output pdf is
        # computable
        # assert self._action_dim == self._sample_dim

        # create network
        self._network = MLP(
            input_shape=(self._obs_dim + self._sample_dim,),
            output_dim=self._action_dim,
            hidden_sizes=hidden_dims,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
        )

        self._l_output = self._network.output_layer
        self._input_var = self._network.input_layer.input_var
        self._sample_var = ext.new_tensor(
            'sample',
            ndim=2,
            dtype=theano.config.floatX
        )

        LasagnePowered.__init__(self, [self._l_output])
        StochasticPolicy.__init__(self, env_spec)
        self._action_var = L.get_output(self._l_output)

        self._f_output = ext.compile_function(
            inputs=[self._input_var],
            outputs=self._action_var,
        )

    def _get_input_samples(self, N):
        """
        Samples from input distribution q_0. Hardcoded as standard normal.

        :param N: Number of samples to be returned.
        :return: A numpy array holding N samples.
        """
        if self._freeze_samples:
            indices = np.random.randint(low=0, high=self._K, size=N)
            samples = self._samples[indices]
            return samples
        else:
            return np.random.randn(N, self._sample_dim)

    @overrides
    def get_action(self, observation, sample=None):
        flat_obs = self.observation_space.flatten(observation)
        if sample is None:
            sample = self._get_input_samples(1)[0]
        inputs = np.concatenate([flat_obs, sample]).reshape(1, -1)
        action = self._f_output(inputs)[0]
        return action, dict()

    def get_actions(self, observations, samples=None):
        flat_obs = self.observation_space.flatten_n(observations)
        N = flat_obs.shape[0]
        if samples is None:
            samples = self._get_input_samples(N)
        inputs = np.concatenate([flat_obs, samples], axis=1)
        actions = self._f_output([inputs])
        return actions, dict()

    @overrides
    def log_diagnostics(self, paths):
        pass

    @property
    def distribution(self):
        return None

    def reset(self):
        pass

    @staticmethod
    def load_from_file(file_name):
        return joblib.load(file_name)

    @staticmethod
    def copy_from_tf_policy(env_spec, tf_policy):
        assert isinstance(tf_policy, TfStochasticNNPolicy)

        def convert_nonlinearity(tf_nonlinearity):
            if tf_nonlinearity == tf.identity:
                return None
            elif tf_nonlinearity == tf.tanh:
                return NL.tanh
            elif tf_nonlinearity == tf.relu:
                return NL.rectify

        # initialize with the same structure
        policy = StochasticNNPolicy(
            env_spec=env_spec,
            hidden_dims=tf_policy._hidden_dims,
            hidden_nonlinearity=NL.rectify, # not rigorous
            output_nonlinearity=convert_nonlinearity(
                tf_policy._output_nonlinearity
            ),
            sample_dim=tf_policy._sample_dim,
            freeze_samples=tf_policy._freeze,
            K=tf_policy._K,
        )
        policy._samples = tf_policy._samples

        # copy the weights and biases
        L.set_all_param_values(
            policy._l_output,
            [p.eval() for p in tf_policy.get_params()],
        )

        return policy

class PolicyCopier(object):
    def __init__(self, retrainer, file_name):
        self.retrainer = retrainer
        self.file_name = file_name

    def run(self):
        env = self.retrainer.get_env()
        tf_policy = self.retrainer.get_policy()
        policy = StochasticNNPolicy.copy_from_tf_policy(
            env_spec=env.spec,
            tf_policy=tf_policy,
        )
        joblib.dump(policy, self.file_name)


# ------------------------------------------------------------------------
def test():
    from sandbox.haoran.myscripts.retrainer import Retrainer
    from sandbox.haoran.myscripts.envs import EnvChooser
    from rllab.envs.normalized_env import normalize
    import time

    env_name = "tuomas_ant"
    env_kwargs = {}
    env_chooser = EnvChooser()
    env = normalize(
        env_chooser.choose_env(env_name,**env_kwargs),
        clip=True,
    )

    retrainer = Retrainer(
        exp_prefix="tuomas/vddpg/",
        exp_name="exp-000_20170212_232051_458883_tuomas_ant",
        snapshot_file="itr_399.pkl",
        configure_script="",
    )
    tf_policy = retrainer.get_policy()
    policy = StochasticNNPolicy.copy_from_tf_policy(
        env_spec=env.spec,
        tf_policy=tf_policy,
    )

    animated = True
    speedup = 10
    max_path_length = 500

    while True:
        o = env.reset()
        policy.reset()
        t = 0
        if animated:
            env.render()
        while t < max_path_length:
            a, agent_info = policy.get_action(o)
            next_o, r, d, env_info = env.step(a)
            t += 1
            if d:
                break
            o = next_o
            if animated:
                env.render()
                timestep = 0.05
                time.sleep(timestep / speedup)
        if animated:
            env.render(close=True)

if __name__ == "__main__":
    test()
