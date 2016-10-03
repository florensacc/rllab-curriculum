import numpy as np
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import theano
import theano.tensor as TT
from rllab.misc.ext import compile_function
from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import ConvNetwork
from rllab.misc import tensor_utils
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.core.serializable import Serializable
from rllab.misc.ext import iterate_minibatches_generic
from rllab.misc import logger
from rllab.core.serializable import Serializable
from rllab.regressors.gaussian_conv_regressor import GaussianConvRegressor

from sandbox.adam.parallel.conjugate_gradient_optimizer import ParallelConjugateGradientOptimizer
from sandbox.adam.parallel.util import SimpleContainer
import multiprocessing as mp

class ParallelGaussianConvRegressor(GaussianConvRegressor):
    """
    A parallel version of GaussianConvRegressor
    Each worker has one copy of the network (could move the weight params to shared memory).
    Prediction is performed individually.
    The only change is that input / output normalization will be synchronized among workers.
    """

    def __init__(
            self,
            **kwargs
    ):
        Serializable.quick_init(self,locals())
        if kwargs["optimizer"] is None:
            kwargs["optimizer"] = ParallelConjugateGradientOptimizer(
                cg_iters=0,
                subsample_factor=1.,
                max_backtracks=0,
                accept_violation=False,
                num_slices=1,
            )
        super(ParallelGaussianConvRegressor,self).__init__(**kwargs)

    def init_rank(self, rank):
        self.rank = rank
        self._optimizer.init_rank(rank)

    def init_par_objs(self, n_parallel):
        par_data = SimpleContainer(
            rank=None,
            n_steps_collected=mp.RawArray('i', n_parallel),
        )
        barriers = SimpleContainer()
        if self._normalize_inputs:
            input_dim = np.prod(self.input_shape)
            par_data.append(
                x_sum_2d=np.reshape(mp.RawArray('d',int(input_dim * n_parallel)),
                (input_dim, n_parallel)),
                x_square_sum_2d=np.reshape(mp.RawArray('d',int(input_dim * n_parallel)),
                (input_dim, n_parallel)),
            )
            barriers.append(normalize_inputs=[mp.Barrier(n_parallel), mp.Barrier(n_parallel)])

        if self._normalize_outputs:
            par_data.append(
                y_sum_2d=np.reshape(mp.RawArray('d',int(self.output_dim * n_parallel)),
                (self.output_dim, n_parallel)),
                y_square_sum_2d=np.reshape(mp.RawArray('d',int(self.output_dim * n_parallel)),
                (self.output_dim, n_parallel)),
            )
            barriers.append(normalize_outputs=[mp.Barrier(n_parallel), mp.Barrier(n_parallel)])

        self._par_objs = (par_data, barriers)

        self._optimizer.init_par_objs(n_parallel,            size_grad=len(self.get_param_values(trainable=True)),
        )

    def force_compile(self):
        xs = np.zeros((1,np.prod(self.input_shape))).astype(theano.config.floatX)
        ys = np.zeros((1,self.output_dim)).astype(theano.config.floatX)
        old_means = np.zeros_like(ys).astype(theano.config.floatX)
        old_log_stds = np.ones_like(ys).astype(theano.config.floatX)
        inputs = (xs,ys,old_means,old_log_stds)
        self._optimizer.force_compile(inputs)

    def fit(self, xs, ys):
        shareds, barriers = self._par_objs

        # report sample sizes
        n_steps_collected = xs.shape[0]
        shareds.n_steps_collected[self.rank] = n_steps_collected
        self._optimizer.set_avg_fac(n_steps_collected)
        total_n_steps_collected = np.sum(shareds.n_steps_collected)

        if self._subsample_factor < 1:
            idx = np.random.randint(0, n_steps_collected, int(n_steps_collected * self._subsample_factor))
            xs, ys = xs[idx], ys[idx]


        """ the only change in parallel version """
        if self._normalize_inputs:
            # each worker computes its statistics
            input_dim = np.prod(self.input_shape)
            x_sum = np.sum(xs, axis=0, keepdims=False).reshape(input_dim)
            x_square_sum = np.sum(xs**2, axis=0, keepdims=False).reshape(input_dim)
            shareds.x_sum_2d[:,self.rank] = x_sum
            shareds.x_square_sum_2d[:,self.rank] = x_square_sum
            barriers.normalize_inputs[0].wait()

            # sum up statistics from different workers
            # currently this is performed by all workers
            x_mean =np.sum(shareds.x_sum_2d,axis=1) / total_n_steps_collected
            x_square_mean = np.sum(shareds.x_square_sum_2d, axis=1) / total_n_steps_collected
            x_std = np.sqrt(x_square_mean - x_mean ** 2 + 1e-8)

            # prepare for NN to use
            self._x_mean_var.set_value(x_mean.reshape((1,-1)).astype(theano.config.floatX))
            self._x_std_var.set_value(x_std.reshape((1,-1)).astype(theano.config.floatX))
            barriers.normalize_inputs[1].wait()


        if self._normalize_outputs:
            # each worker computes its statistics
            output_dim = self.output_dim
            y_sum = np.sum(ys, axis=0, keepdims=False).reshape(output_dim)
            y_square_sum = np.sum(ys**2, axis=0, keepdims=False).reshape(output_dim)
            shareds.y_sum_2d[:,self.rank] = y_sum
            shareds.y_square_sum_2d[:,self.rank] = y_square_sum
            barriers.normalize_outputs[0].wait()

            # sum up statistics from different workers
            # currently this is performed by all workers
            y_mean =np.sum(shareds.y_sum_2d,axis=1) / total_n_steps_collected
            y_square_mean = np.sum(shareds.y_square_sum_2d, axis=1) / total_n_steps_collected
            y_std = np.sqrt(y_square_mean - y_mean ** 2 + 1e-8)

            # prepare for NN to use
            self._y_mean_var.set_value(y_mean.reshape((1,-1)).astype(theano.config.floatX))
            self._y_std_var.set_value(y_std.reshape((1,-1)).astype(theano.config.floatX))
            # DEBUG: check whether the normalization is correct
            # print(y_sum / n_steps_collected, y_mean,np.std(ys), y_std)
            barriers.normalize_outputs[1].wait()

        """"""""""""""""""""""""""""""""""""


        if self._name:
            prefix = self._name + "_"
        else:
            prefix = ""

        # FIXME: needs batch computation to avoid OOM.
        loss_before, loss_after, mean_kl, batch_count = 0., 0., 0., 0
        for batch in iterate_minibatches_generic(input_lst=[xs, ys], batchsize=self._batchsize, shuffle=True):
            batch_count += 1
            _xs, _ys = batch
            if self._use_trust_region:
                old_means, old_log_stds = self._f_pdists(_xs)
                inputs = [_xs, _ys, old_means, old_log_stds]
            else:
                inputs = [_xs, _ys]
            loss_before += self._optimizer._loss(inputs,extra_inputs=None)

            self._optimizer.optimize(inputs)
            loss_after += self._optimizer._loss(inputs,extra_inputs=None)
            if self._use_trust_region:
                mean_kl += self._optimizer._constraint_val(inputs,extra_inputs=None)

        logger.record_tabular(prefix + 'LossBefore', loss_before / batch_count)
        logger.record_tabular(prefix + 'LossAfter', loss_after / batch_count)
        logger.record_tabular(prefix + 'dLoss', (loss_before - loss_after) / batch_count)
        if self._use_trust_region:
            logger.record_tabular(prefix + 'MeanKL', mean_kl / batch_count)
