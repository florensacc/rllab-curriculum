from __future__ import print_function
from __future__ import absolute_import

from rllab.policies.base import StochasticPolicy
from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.misc import tensor_utils
from rllab.misc import ext
from rllab.misc import logger
import theano
import theano.tensor as TT
import numpy as np

floatX = np.cast[theano.config.floatX]


class BayesianNNPolicy(StochasticPolicy, Serializable):
    """
    This class can be used as a wrapper around any parametrized policy, turning it into a policy where we maintain a
    factored Gaussian distribution over its parameters. Gradient computation on the original policy parameters now
    becomes computations on the mean and variance parameters of this distribution. In reality, we parametrize the
    mean and log std of this factored Gaussian distribution.

    This wrapper assumes that the policy does not apply dropout or batch normalization.
    """

    def __init__(self, wrapped_policy, std_mult=1., fixed_std=None, std_type="exp", bbox_grad='none'):
        """
        :param wrapped_policy: policy to apply the wrapper with
        :type wrapped_policy: StochasticPolicy
        :param std_mult: multiplier applied to the initial standard deviation. We compute the standard deviation by
        computing the standard deviation of all entries of a single parameter tensor, and then multiply it by this
        supplied parameter. The motivation is that we usually initialize the parameter from a Gaussian distribution
        with a standard deviation depending on the fan-in of the layer. In the event that the computed standard
        deviation is zero, we set the standard deviation to be equal to std_mult (this could happen to e.g. bias parameter
        values).
        :param fixed_std: whether to have a fixed initial standard deviation over all policy parameters. By default
        this is None, and a parameter-dependent initial standard deviation is chosen.
        :param std_type: how to parametrize the standard deviation of the policy. Can be one of 'exp', 'softplus',
        and 'identity'
        :param bbox_grad: whether to use blackbox gradients for the parameters. Can be one of 'none', 'std', and 'all'
        """
        Serializable.quick_init(self, locals())
        super(BayesianNNPolicy, self).__init__(wrapped_policy._env_spec)
        self._wrapped_policy = wrapped_policy
        self.n_params = len(wrapped_policy.get_param_values())
        self.mean_var = theano.shared(
            floatX(wrapped_policy.get_param_values()),
            name="bnn_mean",
        )
        init_std = []
        for param in wrapped_policy.get_params():
            param_val = param.get_value()
            param_std = np.std(param_val)
            if param_std < 1e-8:
                # if the initial parameter std is close to 0, set the standard deviation to be one (x std_mult)
                param_std = 1.
            init_std.append((np.ones_like(param_val) * param_std * std_mult).flatten())
        std = np.concatenate(init_std)
        self.std_type = std_type
        self.bbox_grad = bbox_grad
        if std_type == "exp":
            self.std_param_var = theano.shared(
                floatX(np.log(std)),
                name="bnn_std_param"
            )
        elif std_type == "identity":
            self.std_param_var = theano.shared(
                floatX(std),
                name="bnn_std_param"
            )
        elif std_type == "softplus":
            # std = log(1 + e^(param))
            # => param = log(exp(std) - 1)
            self.std_param_var = theano.shared(
                floatX(np.log(np.exp(std) - 1)),
                name="bnn_std_param"
            )
        else:
            raise NotImplementedError
        self.param_epsilon_var = theano.shared(
            floatX(np.zeros((self.n_params,))),
            name="param_epsilon"
        )
        self.reset()

    @property
    def wrapped_policy(self):
        return self._wrapped_policy

    def _get_std(self):
        std_param = self.std_param_var.get_value()
        if self.std_type == "exp":
            std = np.exp(std_param)
        elif self.std_type == "softplus":
            std = np.log(np.exp(std_param) + 1)
        elif self.std_type == "identity":
            std = std_param
        else:
            raise NotImplementedError
        return std

    def param_log_likelihood_sym(self, param_var):
        if self.std_type == "exp":
            log_std_var = self.std_param_var
        elif self.std_type == "identity":
            log_std_var = TT.log(TT.abs_(self.std_param_var) + 1e-8)
        elif self.std_type == "softplus":
            log_std_var = TT.log(TT.log(1. + TT.exp(self.std_param_var)))
        else:
            raise NotImplementedError
        return DiagonalGaussian(self.n_params).log_likelihood_sym(
            param_var, dict(mean=self.mean_var, log_std=log_std_var)
        )

    def reset(self):
        mean = self.mean_var.get_value()
        std = self._get_std()
        epsilon = np.random.standard_normal((self.n_params,))
        epsilon = np.cast['float32'](epsilon)
        param_val = mean + np.maximum(0, std) * epsilon
        self.param_epsilon_var.set_value(epsilon)
        self.wrapped_policy.set_param_values(param_val)

    def get_params(self, **tags):
        return [self.mean_var, self.std_param_var]

    def get_action(self, observation):
        return self.wrapped_policy.get_action(observation)

    # it is assumed that dist_info_sym will be computed only for a single given path, where we only have a fixed
    # value for param_epsilon
    def dist_info_sym(self, obs_var, state_info_vars):
        wrapped_dist_info_sym = self.wrapped_policy.dist_info_sym(obs_var, state_info_vars)
        if self.std_type == "exp":
            std_var = TT.exp(self.std_param_var)
        elif self.std_type == "softplus":
            std_var = TT.log(TT.exp(self.std_param_var) + 1)
        elif self.std_type == "identity":
            std_var = self.std_param_var
        else:
            raise NotImplementedError
        param_var = self.mean_var + TT.maximum(0, std_var) * self.param_epsilon_var
        origi_params = self.wrapped_policy.get_params()
        unflat_params = ext.unflatten_tensor_variables(param_var, self.wrapped_policy.get_param_shapes(), origi_params)
        dict_items = wrapped_dist_info_sym.items()
        cloned = theano.clone([x[1] for x in dict_items], replace=zip(origi_params, unflat_params))
        keys = [x[0] for x in dict_items]
        return dict(zip(keys, cloned))

    def dist_info(self, obs, state_infos):
        # TODO
        raise NotImplementedError

    @property
    def state_info_keys(self):
        return self.wrapped_policy.state_info_keys

    @property
    def distribution(self):
        return self.wrapped_policy.distribution

    def log_diagnostics(self, paths):
        logger.record_tabular("AveragePolicyParamAbsMean", np.mean(np.abs(self.mean_var.get_value())))
        logger.record_tabular("AveragePolicyParamStd", np.mean(np.maximum(0, self._get_std())))
