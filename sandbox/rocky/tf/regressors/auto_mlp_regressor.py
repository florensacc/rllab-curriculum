import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import logger
import numpy as np
import itertools

from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from sandbox.rocky.tf.distributions.product_distribution import ProductDistribution
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.lbfgs_optimizer import LbfgsOptimizer
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.spaces import Discrete, Box, Product


def space_to_distribution(space):
    """
    Build a distribution from the given space
    """
    if isinstance(space, Discrete):
        return Categorical(space.n)
    elif isinstance(space, Box):
        return DiagonalGaussian(space.flat_dim)
    elif isinstance(space, Product):
        components = space.components
        component_dists = list(map(space_to_distribution, components))
        return ProductDistribution(component_dists)
    else:
        raise NotImplementedError


def space_to_dist_dim(space):
    if isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Box):
        return space.flat_dim * 2
    elif isinstance(space, Product):
        components = space.components
        return sum(map(space_to_dist_dim, components))
    else:
        raise NotImplementedError


def output_to_info(output_var, output_space):
    if isinstance(output_space, Discrete):
        return dict(prob=tf.nn.softmax(output_var))
    elif isinstance(output_space, Box):
        mean = output_var[:, :output_space.flat_dim]
        log_std = output_var[:, output_space.flat_dim:]
        return dict(mean=mean, log_std=log_std)
    elif isinstance(output_space, Product):
        components = output_space.components
        dimensions = [space_to_dist_dim(x) for x in components]
        cum_dims = list(np.cumsum(dimensions))
        ret = dict()
        for idx, slice_from, slice_to, subspace in zip(itertools.count(), [0] + cum_dims, cum_dims, components):
            sub_info = output_to_info(output_var[:, slice_from:slice_to], subspace)
            for k, v in sub_info.items():
                ret["id_%d_%s" % (idx, k)] = v
        return ret


class AutoMLPRegressor(LayersPowered, Serializable):
    def __init__(
            self,
            name,
            input_shape,
            output_space,
            network=None,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            optimizer=None,
            use_trust_region=True,
            step_size=0.01,
            normalize_inputs=True,
            separate_networks=True,
    ):
        Serializable.quick_init(self, locals())

        output_dim = output_space.flat_dim

        with tf.variable_scope(name):
            if optimizer is None:
                if use_trust_region:
                    optimizer = PenaltyLbfgsOptimizer("optimizer")
                else:
                    optimizer = LbfgsOptimizer("optimizer")
            self._optimizer = optimizer

            if network is None:
                if separate_networks:
                    if isinstance(output_space, Product) and not any([isinstance(x, Product) for x in
                                                                      output_space.components]):
                        input_layer = L.InputLayer(shape=(None,) + input_shape, name="input")
                        l_outputs = []
                        for idx, component_i in enumerate(output_space.components):
                            network_i = MLP(
                                name="regressor_%d" % idx,
                                input_shape=input_shape,
                                input_layer=input_layer,
                                hidden_sizes=hidden_sizes,
                                hidden_nonlinearity=hidden_nonlinearity,
                                output_dim=space_to_dist_dim(component_i),
                                output_nonlinearity=None,
                            )
                            l_outputs.append(network_i.output_layer)
                        l_output = L.concat(l_outputs)
                        xs_var = input_layer.input_var
                    else:
                        raise NotImplementedError
                else:
                    network = MLP(
                        name="regressor",
                        input_shape=input_shape,
                        output_dim=space_to_dist_dim(output_space),
                        hidden_sizes=hidden_sizes,
                        hidden_nonlinearity=hidden_nonlinearity,
                        # separate nonlinearities will be used for each component of the output
                        output_nonlinearity=None,
                    )

                    l_output = network.output_layer
                    xs_var = network.input_var
                    input_layer = network.input_layer
            else:
                xs_var = network.input_var
                l_output = network.output_layer
                input_layer = network.input_layer

            LayersPowered.__init__(self, [l_output])

            ys_var = tf.placeholder(dtype=tf.float32, shape=(None, output_space.flat_dim), name="ys")

            x_mean_var = tf.Variable(
                initial_value=np.zeros((1,) + input_shape),
                name="x_mean",
                dtype=tf.float32
            )
            x_std_var = tf.Variable(
                initial_value=np.ones((1,) + input_shape),
                name="x_std",
                dtype=tf.float32
            )

            normalized_xs_var = (xs_var - x_mean_var) / x_std_var

            output_var = L.get_output(l_output, {input_layer: normalized_xs_var})

            dist = self._dist = space_to_distribution(output_space)

            info_vars = output_to_info(output_var, output_space)

            old_info_vars_list = [
                tf.placeholder(dtype=tf.float32, shape=shape, name="old_%s" % k)
                for k, shape in dist.dist_info_specs
                ]
            old_info_vars = dict(list(zip(dist.dist_info_keys, old_info_vars_list)))

            mean_kl = tf.reduce_mean(dist.kl_sym(old_info_vars, info_vars))

            loss = - tf.reduce_mean(dist.log_likelihood_sym(ys_var, info_vars))

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
            self._f_dist_info = tensor_utils.compile_function([xs_var], info_vars)

            self._use_trust_region = use_trust_region
            self._name = name

            self._normalize_inputs = normalize_inputs
            self._xs_var = xs_var
            self._x_mean_var = x_mean_var
            self._x_std_var = x_std_var

            xs_mean, xs_variance = tf.nn.moments(xs_var, axes=[0], keep_dims=True)
            self._f_normalize_x = tensor_utils.compile_function([xs_var], tf.group(
                tf.assign(x_mean_var, xs_mean),
                tf.assign(x_std_var, tf.sqrt(xs_variance)),
            ))

    def fit(self, xs, ys):
        if self._normalize_inputs:
            # recompute normalizing constants for inputs
            self._f_normalize_x(xs)
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

    def predict_dist(self, xs):
        return self._f_dist_info(xs)

    def predict_sample(self, xs):
        dist_info = self.predict_dist(xs)
        return self._dist.sample(dist_info)

    def predict_log_likelihood(self, xs, ys):
        dist_info = self.predict_dist(xs)
        return self._dist.log_likelihood(np.asarray(ys), dist_info)
