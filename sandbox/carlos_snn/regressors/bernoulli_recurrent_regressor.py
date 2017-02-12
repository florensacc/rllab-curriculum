import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np
import theano
import theano.tensor as TT

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.distributions.bernoulli import Bernoulli
from rllab.misc import ext
from rllab.misc import logger
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.core.network import GRUNetwork


class BernoulliRecurrentRegressor(LasagnePowered, Serializable):
    """
    A class for performing regression (or classification, really) by fitting a bernoulli distribution to each of the
    output units.
    """

    def __init__(
            self,
            input_shape,
            output_dim,
            predict_all=False,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=NL.rectify,
            optimizer=None,
            use_trust_region=True,
            step_size=0.01,
            normalize_inputs=True,
            name=None,
    ):
        """
        :param input_shape: Shape of the input data.
        :param output_dim: Dimension of output.
        :param predict_all: use the prediction made at every step about the latent variables (not only the last step)
        :param hidden_sizes: Number of hidden units of each layer of the mean network.
        :param hidden_nonlinearity: Non-linearity used for each layer of the mean network.
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

        self.output_dim = output_dim
        self._optimizer = optimizer

        p_network = GRUNetwork(
            input_shape=input_shape,
            output_dim=output_dim,
            hidden_dim=hidden_sizes[0],
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=NL.sigmoid,
        )

        l_p = p_network.output_layer  # this is every intermediate latent state! but I only care about last

        LasagnePowered.__init__(self, [l_p])

        xs_var = p_network.input_layer.input_var

        ys_var = TT.itensor3("ys")  # this is 3D: (traj, time, lat_dim)
        old_p_var = TT.tensor3("old_p")
        x_mean_var = theano.shared(
            np.zeros((1, 1,) + input_shape),
            name="x_mean",
            broadcastable=(True, True,) + (False,) * len(input_shape)
        )

        x_std_var = theano.shared(
            np.ones((1, 1,) + input_shape),
            name="x_std",
            broadcastable=(True, True,) + (False,) * len(input_shape)
        )

        normalized_xs_var = (xs_var - x_mean_var) / x_std_var
# this is the previous p_var, from which I only want the last time-step padded along all time-steps
        p_var_all = L.get_output(l_p, {p_network.input_layer: normalized_xs_var})
# take only last dim but keep the shape
        p_var_last = TT.reshape(p_var_all[:,-1,:],(TT.shape(p_var_all)[0],1,TT.shape(p_var_all)[2]))
# padd along the time dimension to obtain the same shape as before
        padded_p = TT.tile(p_var_last, (1, TT.shape(p_var_all)[1], 1))
# give it the standard name
        if predict_all:
            p_var = p_var_all
        else:
            p_var = padded_p

        old_info_vars = dict(p=old_p_var)
        info_vars = dict(p=p_var)  # posterior of the latent at every step, wrt obs-act. Same along batch if recurrent

        dist = self._dist = Bernoulli(output_dim)

        mean_kl = TT.mean(dist.kl_sym(old_info_vars, info_vars))

        loss = - TT.mean(dist.log_likelihood_sym(ys_var, info_vars)) # regressor just wants to min -loglik of data ys

        predicted = p_var >= 0.5

        self._f_predict = ext.compile_function([xs_var], predicted)
        self._f_p = ext.compile_function([xs_var], p_var)  # for consistency with gauss_mlp_reg this should be ._f_pdists

        self._l_p = l_p

        optimizer_args = dict(
            loss=loss,
            target=self,
            network_outputs=[p_var],
        )

        if use_trust_region:
            optimizer_args["leq_constraint"] = (mean_kl, step_size)
            optimizer_args["inputs"] = [xs_var, ys_var, old_p_var]
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
            self._x_mean_var.set_value(np.mean(xs, axis=(0, 1), keepdims=True)) #the mean taken over batches AND steps
            self._x_std_var.set_value(np.std(xs, axis=(0, 1), keepdims=True) + 1e-8)
        if self._use_trust_region:
            old_p = self._f_p(xs)
            inputs = [xs, ys, old_p]
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
        return self._f_predict(np.asarray(xs))

    def sample_predict(self, xs):
        p = self._f_p(xs)
        return self._dist.sample(dict(p=p))

    def predict_log_likelihood(self, xs, ys):
        p = self._f_p(np.asarray(xs))
        return self._dist.log_likelihood(np.asarray(ys), dict(p=p))

    def get_param_values(self, **tags):
        return LasagnePowered.get_param_values(self, **tags)

    def set_param_values(self, flattened_params, **tags):
        return LasagnePowered.set_param_values(self, flattened_params, **tags)
