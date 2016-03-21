from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.lasagne_layers import ParamLayer
from rllab.core.serializable import Serializable
from rllab.core.network import MLP
from rllab.misc import categorical_dist
from rllab.misc import ext
from rllab.misc import special
from rllab.optimizer.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.optimizer.lbfgs_optimizer import LbfgsOptimizer
from rllab.misc import logger
import theano
import theano.tensor as TT
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import numpy as np

NONE = list()


class CategoricalMLPRegressor(LasagnePowered, Serializable):

    """
    A class for performing regression (or classification, really) by fitting a categorical distribution to the outputs.
    Assumes that the outputs will be always a one hot vector.
    """

    def __init__(
            self,
            input_shape,
            output_dim,
            hidden_sizes=(32, 32),
            nonlinearity=NL.rectify,
            optimizer=None,
            use_trust_region=True,
            step_size=0.01,
            normalize_inputs=True,
            name=None,
    ):
        """
        :param input_shape: Shape of the input data.
        :param output_dim: Dimension of output.
        :param hidden_sizes: Number of hidden units of each layer of the mean network.
        :param nonlinearity: Non-linearity used for each layer of the mean network.
        :param optimizer: Optimizer for minimizing the negative log-likelihood.
        :param use_trust_region: Whether to use trust region constraint.
        :param step_size: KL divergence constraint for each iteration
        """
        Serializable.quick_init(self, locals())

        if optimizer is None:
            if use_trust_region:
                optimizer = PenaltyLbfgsOptimizer()
            else:
                optimizer = LbfgsOptimizer()

        self._optimizer = optimizer

        log_prob_network = MLP(
            input_shape=input_shape,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            nonlinearity=nonlinearity,
            output_nonlinearity=theano.tensor.nnet.logsoftmax
        )

        l_log_prob = log_prob_network.output_layer

        LasagnePowered.__init__(self, [l_log_prob])

        xs_var = log_prob_network.input_layer.input_var
        ys_var = TT.matrix("ys")
        old_log_prob_var = TT.matrix("old_log_prob")

        x_mean_var = theano.shared(
            np.zeros((1,) + input_shape),
            name="x_mean",
            broadcastable=(True,) + (False, ) * len(input_shape)
        )
        x_std_var = theano.shared(
            np.ones((1,) + input_shape),
            name="x_std",
            broadcastable=(True,) + (False, ) * len(input_shape)
        )

        normalized_xs_var = (xs_var - x_mean_var) / x_std_var

        log_prob_var = L.get_output(l_log_prob, {log_prob_network.input_layer: normalized_xs_var})

        mean_kl = TT.mean(categorical_dist.kl_sym(old_log_prob_var, log_prob_var))

        loss = - TT.mean(categorical_dist.log_likelihood_sym(ys_var, log_prob_var))

        predicted = special.to_onehot_sym(TT.argmax(log_prob_var, axis=1), output_dim)

        self._f_predict = ext.compile_function([xs_var], predicted)
        self._f_log_prob = ext.compile_function([xs_var], log_prob_var)
        self._l_log_prob = l_log_prob

        optimizer_args = dict(
            loss=loss,
            target=self,
            network_outputs=[log_prob_var],
        )

        if use_trust_region:
            optimizer_args["leq_constraint"] = (mean_kl, step_size)
            optimizer_args["inputs"] = [xs_var, ys_var, old_log_prob_var]
        else:
            optimizer_args["inputs"] = [xs_var, ys_var]

        self._optimizer.update_opt(**optimizer_args)

        self._use_trust_region = use_trust_region
        self._name = name

        self._normalize_inputs = normalize_inputs
        self._x_mean_var = x_mean_var
        self._x_std_var = x_std_var

    def fit(self, xs, ys):
        if self._normalize_inputs:
            # recompute normalizing constants for inputs
            self._x_mean_var.set_value(np.mean(xs, axis=0, keepdims=True))
            self._x_std_var.set_value(np.std(xs, axis=0, keepdims=True) + 1e-8)
        if self._use_trust_region:
            old_log_prob = self._f_log_prob(xs)
            inputs = [xs, ys, old_log_prob]
        else:
            inputs = [xs, ys]
        loss_before = self._optimizer.loss(inputs)
        if self._name:
            prefix = self._name + "_"
        else:
            prefix = ""
        logger.record_tabular(prefix + 'LossBefore', loss_before)
        self._optimizer.optimize(inputs)
        loss_after = self._optimizer.loss(inputs)
        logger.record_tabular(prefix + 'LossAfter', loss_after)
        logger.record_tabular(prefix + 'dLoss', loss_before - loss_after)

    def predict(self, xs):
        return self._f_predict(xs)

    def predict_log_likelihood(self, xs, ys):
        log_prob = self._f_log_prob(xs)
        return categorical_dist.log_likelihood(ys, log_prob)

    def get_param_values(self, **tags):
        return LasagnePowered.get_param_values(self, **tags)

    def set_param_values(self, flattened_params, **tags):
        return LasagnePowered.set_param_values(self, flattened_params, **tags)
