from rllab.misc import ext
from rllab.misc import logger
from rllab.core.serializable import Serializable
import time
import lasagne.updates
import theano
import pyprind
import numpy as np
import multiprocessing as mp

from sandbox.adam.parallel.util import SimpleContainer


class ParallelFirstOrderOptimizer(Serializable):
    """
    Synchronized parallel gradient descent

    Each process collects a mini-batch and computes the gradient. Then the main
        process averages the gradient steps and update the params. One update
        is one epoch. After each update, the total loss is computed over all
        data and reported to the terminal.
    """

    def __init__(
            self,
            name="",
            update_method="sgd",
            learning_rate=1e-3,
            max_epochs=1,
            tolerance=1e-6,
            batch_size=None,
            callback=None,
            verbose=False,
            num_slices=1,
            **kwargs):
        """
        :param max_epochs: number of testing / logging epochs
        :param tolerance: exit an epoch if |loss - new_loss| < tolerance
        :param batch_size: within an epoch, each iteration samples a minibatch (shuffle first and then read sequentially) and do gradient descent; "None" means using all data to compute a gradient step
        :param callback: log info for each epoch
        :param num_slices: divide the dataset and aggregate the results, useful if want to use a huge batch size for a single gradient step
        :return:
        """
        Serializable.quick_init(self, locals())
        self._name = name
        self._opt_fun = None
        self._target = None
        self._lr = learning_rate
        self._tolerance = tolerance
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._callback = callback
        self._update_method = update_method
        self._verbose = verbose
        self._num_slices = num_slices

    def init_rank(self,rank):
        self.rank = rank
        self.vb = self.pd.vb

    def init_par_objs(self, n_parallel, size_grad):
        n_grad_elm_worker = -(-size_grad // n_parallel)  # ceiling div
        vb_idx = [n_grad_elm_worker * i for i in range(n_parallel + 1)]
        vb_idx[-1] = size_grad

        par_data = SimpleContainer(
            rank=None,
            avg_fac=1.0 / n_parallel,
            vb=[(vb_idx[i], vb_idx[i + 1]) for i in range(n_parallel)],
        )
        self.pd = par_data

        shareds = SimpleContainer(
            flat_g=np.frombuffer(mp.RawArray('d', size_grad)),
            grads_2d=np.reshape(
                np.frombuffer(mp.RawArray('d', size_grad * n_parallel)),
                (size_grad, n_parallel)),
            loss=mp.RawArray('d', n_parallel),
            n_steps_collected=mp.RawArray('i', n_parallel),
            cur_param=np.frombuffer(mp.RawArray('d', size_grad)),
        )
        barriers = SimpleContainer(
            avg_fac=mp.Barrier(n_parallel),
            flat_g=[mp.Barrier(n_parallel) for _ in range(2)],
            loss=mp.Barrier(n_parallel),
            update=mp.Barrier(n_parallel),
            sync_param=[mp.Barrier(n_parallel) for _ in range(2)],
        )
        self._par_objs = (shareds, barriers)


    def force_compile(self, inputs, extra_inputs=None):
        """
        Serial - force compiling of Theano functions.
        """
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        self._opt_fun["f_loss"](*(inputs + extra_inputs))
        x = self._opt_fun["f_grad"](*(inputs + extra_inputs))

    def set_avg_fac(self, n_steps_collected):
        shareds, barriers = self._par_objs
        shareds.n_steps_collected[self.rank] = n_steps_collected
        barriers.avg_fac.wait()
        self.avg_fac = 1.0 * n_steps_collected / sum(shareds.n_steps_collected)

    def update_opt(self, loss, target, inputs, extra_inputs=None, gradients=None, **kwargs):
        self._target = target

        if gradients is None:
            gradients = theano.grad(loss, target.get_params(trainable=True), disconnected_inputs='ignore')
        flat_grad = ext.flatten_tensor_variables(gradients)

        if extra_inputs is None:
            extra_inputs = list()

        self._opt_fun = ext.lazydict(
            f_loss=lambda: ext.compile_function(
                inputs + extra_inputs,
                loss,
                log_name=self._name + "_f_loss"
            ),
            f_grad=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=flat_grad,
                log_name=self._name + "_f_grad"
            )
        )

    def loss(self, inputs, extra_inputs=None):
        shareds, barriers = self._par_objs
        shareds.loss[self.rank] = self.avg_fac * ext.sliced_fun(
            self._opt_fun["f_loss"], self._num_slices)(inputs, extra_inputs)
        barriers.loss.wait()
        return sum(shareds.loss)

    def _loss(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    def flat_g(self, inputs, extra_inputs=None):
        shareds, barriers = self._par_objs
        # Each worker records result available to all.
        shareds.grads_2d[:, self.rank] = self.avg_fac * ext.sliced_fun(
            self._opt_fun["f_grad"], self._num_slices)(inputs, extra_inputs)
        barriers.flat_g[0].wait()
        if self.rank == 0:
            shareds.flat_g = np.sum(shareds.grads_2d, axis=1)
        barriers.flat_g[1].wait()
        return shareds.flat_g

    def synchronize_param(self):
        """ copy params from master process to others """
        shareds, barriers = self._par_objs
        if self.rank == 0:
            shareds.cur_param[:] = self._target.get_param_values(trainable=True)
        barriers.sync_param[0].wait()
        # WARNING: must use array[:] = ... to assign values to shared memory; otherwise it doesn't happen. See below for debugging.
        # print(self.rank, hex(id(shareds.cur_param)), shareds.cur_param)
        self._target.set_param_values(shareds.cur_param,trainable=True)
        barriers.sync_param[1].wait()

    def optimize_gen(self, inputs, extra_inputs=None, callback=None, yield_itr=None):
        shareds, barriers = self._par_objs
        if len(inputs) == 0:
            # Assumes that we should always sample mini-batches
            raise NotImplementedError

        if extra_inputs is None:
            extra_inputs = tuple()

        last_loss = self.loss(inputs, extra_inputs)

        start_time = time.time()

        dataset = BatchDataset(
            inputs, self._batch_size,
            extra_inputs=extra_inputs
            #, randomized=self._randomized
        )

        itr = 0
        prev_params = self._target.get_param_values(trainable=True)
        if self.rank == 0 and self._verbose:
            epochs = pyprind.prog_bar(list(range(self._max_epochs)))
        else:
            epochs = range(self._max_epochs)

        for epoch in epochs:
            for _inputs, _extra_inputs in dataset.iterate(update=False):
                flat_g = self.flat_g(_inputs, _extra_inputs)
                if self.rank == 0:
                    if self._update_method == "sgd":
                        descent_step = -self._lr * flat_g
                    else:
                        #TODO: implement Adam, Adagrad, etc.
                        raise NotImplementedError
                    self._target.set_param_values(
                        self._target.get_param_values(trainable=True) + descent_step,
                        trainable=True,
                    )

                self.synchronize_param()
                if yield_itr is not None and (itr % (yield_itr+1)) == 0:
                    yield
                itr += 1

            new_loss = self.loss(inputs, extra_inputs)

            # logging
            if self.rank == 0:
                if self._verbose:
                    logger.log("Epoch %d, loss %s" % (epoch, new_loss))

                if self._callback or callback:
                    elapsed = time.time() - start_time
                    callback_args = dict(
                        loss=new_loss,
                        params=self._target.get_param_values(trainable=True) if self._target else None,
                        itr=epoch,
                        elapsed=elapsed,
                    )
                    if self._callback:
                        self._callback(callback_args)
                    if callback:
                        callback(**callback_args)

            if abs(last_loss - new_loss) < self._tolerance:
                break
            last_loss = new_loss

    def optimize(self, inputs, **kwargs):
        for _ in self.optimize_gen(inputs, **kwargs):
            pass


class BatchDataset(object):
    """
    Copied from rllab.optimizers.minibatch_dataset.
    Only difference: output inputs and extra_inputs separately.
    """

    def __init__(self, inputs, batch_size, extra_inputs=None):
        self._inputs = inputs
        if extra_inputs is None:
            extra_inputs = tuple()
        self._extra_inputs = extra_inputs
        self._batch_size = batch_size
        if batch_size is not None:
            self._ids = np.arange(self._inputs[0].shape[0])
            self.update()

    @property
    def number_batches(self):
        if self._batch_size is None:
            return 1
        return int(np.ceil(self._inputs[0].shape[0] * 1.0 / self._batch_size))

    def iterate(self, update=True):
        if self._batch_size is None:
            yield self._inputs, self._extra_inputs
        else:
            for itr in range(self.number_batches):
                batch_start = itr * self._batch_size
                batch_end = (itr + 1) * self._batch_size
                batch_ids = self._ids[batch_start:batch_end]
                batch = tuple([d[batch_ids] for d in self._inputs])
                yield batch, self._extra_inputs
            if update:
                self.update()

    def update(self):
        np.random.shuffle(self._ids)
