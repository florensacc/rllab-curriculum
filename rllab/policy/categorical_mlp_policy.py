import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc.special import weighted_sample, to_onehot
from rllab.misc.ext import compile_function
from rllab.policy.base import StochasticPolicy
from rllab.misc import categorical_dist


class CategoricalMLPPolicy(StochasticPolicy, LasagnePowered, Serializable):

    def __init__(
            self,
            mdp_spec,
            hidden_sizes=(32, 32),
            nonlinearity='lasagne.nonlinearities.rectify'):
        """
        :param mdp_spec: A spec for the mdp.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        Serializable.quick_init(self, locals())

        prob_network = MLP(
            input_shape=mdp_spec.observation_shape,
            output_dim=mdp_spec.action_dim,
            hidden_sizes=hidden_sizes,
            nonlinearity=nonlinearity,
            output_nl=NL.softmax,
        )

        self._l_prob = prob_network.l_out
        self._l_obs = prob_network.l_in
        self._f_prob = compile_function([prob_network.input_var], L.get_output(prob_network.l_out))

        super(CategoricalMLPPolicy, self).__init__(mdp_spec)
        LasagnePowered.__init__(self, [prob_network.l_out])

    @overrides
    def get_pdist_sym(self, obs_var):
        return L.get_output(self._l_prob, {self._l_obs: obs_var})

    @overrides
    def kl(self, old_prob_var, new_prob_var):
        return categorical_dist.kl_sym(old_prob_var, new_prob_var)

    @overrides
    def likelihood_ratio(self, old_prob_var, new_prob_var, action_var):
        return categorical_dist.likelihood_ratio_sym(
            action_var, old_prob_var, new_prob_var)

    @overrides
    def compute_entropy(self, pdist):
        return np.mean(categorical_dist.entropy(pdist))

    def get_pdists(self, observations):
        return self._f_prob(observations)

    @property
    @overrides
    def pdist_dim(self):
        return self.action_dim

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        prob = self._f_prob([observation])[0]
        action = weighted_sample(prob, xrange(self.action_dim))
        return to_onehot(action, self.action_dim), prob
