from __future__ import print_function
from __future__ import absolute_import
import lasagne.layers as L
import lasagne.nonlinearities as LN
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.core.serializable import Serializable
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from rllab.distributions.categorical import Categorical
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.misc import ext
from rllab.misc import logger
from sandbox.rocky.hrl.distributions.product_distribution import ProductDistribution
import theano
import theano.tensor as TT
import numpy as np
import itertools


def space_to_distribution(space):
    """
    Build a distribution from the given space
    """
    if isinstance(space, Discrete):
        return Categorical()
    elif isinstance(space, Box):
        return DiagonalGaussian()
    elif isinstance(space, Product):
        components = space.components
        component_dists = map(space_to_distribution, components)
        dimensions = [x.flat_dim for x in components]
        return ProductDistribution(component_dists, dimensions)
    else:
        raise NotImplementedError


def output_to_info(output_var, output_space):
    if isinstance(output_space, Discrete):
        return dict(prob=TT.nnet.softmax(output_var))
    elif isinstance(output_space, Box):
        raise NotImplementedError
    elif isinstance(output_space, Product):
        components = output_space.components
        dimensions = [x.flat_dim for x in components]
        cum_dims = list(np.cumsum(dimensions))
        ret = dict()
        for idx, slice_from, slice_to, subspace in zip(itertools.count(), [0] + cum_dims, cum_dims, components):
            sub_info = output_to_info(output_var[:, slice_from:slice_to], subspace)
            for k, v in sub_info.iteritems():
                ret["id_%d_%s" % (idx, k)] = v
        return ret


class SharedNetworkAutoMLPRegressor(LasagnePowered, Serializable):
    def __init__(
            self,
            input_shape,
            output_dim,
            output_space,
            network=None,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=LN.rectify,
            optimizer=None,
            use_trust_region=True,
            step_size=0.01,
            normalize_inputs=True,
            name=None,
    ):
        Serializable.quick_init(self, locals())

        if optimizer is None:
            if use_trust_region:
                optimizer = PenaltyLbfgsOptimizer()
            else:
                optimizer = LbfgsOptimizer()
        self._optimizer = optimizer

        if network is None:
            network = MLP(
                input_shape=input_shape,
                output_dim=output_space.flat_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                # separate nonlinearities will be used for each component of the output
                output_nonlinearity=None,
            )

        l_output = network.output_layer
        LasagnePowered.__init__(self, [l_output])

        xs_var = network.input_var
        ys_var = TT.matrix("ys")

        x_mean_var = theano.shared(
            np.zeros((1,) + input_shape),
            name="x_mean",
            broadcastable=(True,) + (False,) * len(input_shape)
        )
        x_std_var = theano.shared(
            np.ones((1,) + input_shape),
            name="x_std",
            broadcastable=(True,) + (False,) * len(input_shape)
        )

        normalized_xs_var = (xs_var - x_mean_var) / x_std_var

        output_var = L.get_output(l_output, {network.input_layer: normalized_xs_var})

        dist = self._dist = space_to_distribution(output_space)

        info_vars = output_to_info(output_var, output_space)

        old_info_vars_list = [TT.matrix("old_%s" % k) for k in dist.dist_info_keys]
        old_info_vars = dict(zip(dist.dist_info_keys, old_info_vars_list))

        mean_kl = TT.mean(dist.kl_sym(old_info_vars, info_vars))

        loss = - TT.mean(dist.log_likelihood_sym(ys_var, info_vars))

        optimizer_args = dict(
            loss=loss,
            target=self,
            network_outputs=[output_var],
        )

        if use_trust_region:
            optimizer_args["leq_constraint"] = (mean_kl, step_size)
            optimizer_args["inputs"] = [xs_var, ys_var] + old_info_vars_list
        else:
            optimizer_args["inputs"] = [xs_var, ys_var]

        self._optimizer.update_opt(**optimizer_args)
        self._f_dist_info = ext.compile_function([xs_var], info_vars)

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
            old_infos = self._f_dist_info(xs)
            old_info_list = [old_infos[k] for k in self._dist.dist_info_keys]
            inputs = [xs, ys] + old_info_list
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
        raise NotImplementedError

    def predict_log_likelihood(self, xs, ys):
        dist_info = self._f_dist_info(xs)
        return self._dist.log_likelihood(np.asarray(ys), dist_info)

    def get_param_values(self, **tags):
        return LasagnePowered.get_param_values(self, **tags)

    def set_param_values(self, flattened_params, **tags):
        return LasagnePowered.set_param_values(self, flattened_params, **tags)
