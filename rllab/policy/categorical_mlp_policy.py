import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np
import theano.tensor.nnet
from rllab.spaces import Discrete
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import ext
from rllab.policy.base import StochasticPolicy
from rllab.misc import categorical_dist


class CategoricalMLPPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            hidden_sizes=(32, 32),
            nonlinearity=NL.rectify):
        """
        :param env_spec: A spec for the mdp.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Discrete)

        log_prob_network = MLP(
            input_shape=(env_spec.observation_space.flat_dim,),
            output_dim=env_spec.action_space.n,
            hidden_sizes=hidden_sizes,
            nonlinearity=nonlinearity,
            output_nonlinearity=theano.tensor.nnet.logsoftmax,
        )

        self._l_log_prob = log_prob_network.output_layer
        self._l_obs = log_prob_network.input_layer
        self._f_log_prob = ext.compile_function([log_prob_network.input_layer.input_var], L.get_output(
            log_prob_network.output_layer))

        super(CategoricalMLPPolicy, self).__init__(env_spec)
        LasagnePowered.__init__(self, [log_prob_network.output_layer])

    @overrides
    def info_sym(self, obs_var, action_var):
        return dict(log_prob=L.get_output(self._l_log_prob, {self._l_obs: obs_var}))

    @overrides
    def kl_sym(self, old_info_vars, new_info_vars):
        return categorical_dist.kl_sym(old_info_vars, new_info_vars)

    @overrides
    def likelihood_ratio_sym(self, action_var, old_info_vars, new_info_vars):
        return categorical_dist.likelihood_ratio_sym(
            action_var, old_info_vars, new_info_vars)

    @overrides
    def log_likelihood_sym(self, obs_var, action_var):
        info_vars = self.info_sym(obs_var, action_var)
        return categorical_dist.log_likelihood_sym(action_var, info_vars)

    @overrides
    def entropy(self, info):
        return np.mean(categorical_dist.entropy(info))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def act(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        log_prob = self._f_log_prob([flat_obs])[0]
        action = self.action_space.weighted_sample(np.exp(log_prob))
        return action, dict(log_prob=log_prob)

    @property
    def info_keys(self):
        return ["log_prob"]

    @property
    def dist_family(self):
        return categorical_dist
