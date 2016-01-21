from pydoc import locate

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
from rllab.misc.ext import compile_function
from rllab.misc.tensor_utils import flatten_tensors
from rllab.baseline.base import Baseline
from rllab.misc.overrides import overrides
from rllab.sampler import parallel_sampler
import rllab.misc.logger as logger
import numpy as np
import theano
import theano.tensor as TT
import lasagne.layers as L
import lasagne


G = parallel_sampler.G


def worker_init_opt():
    baseline = G.baseline

    new_v_var = TT.vector("new_values")
    loss = TT.mean(TT.square(baseline.v_var - new_v_var[:, np.newaxis]))
    input_list = [baseline.input_var, new_v_var]

    grads = theano.gradient.grad(loss, baseline.get_params(trainable=True))

    G.nn_baseline_opt_info = dict(
        loss=compile_function(input_list, loss),
        grads=compile_function(input_list, grads)
    )


def worker_prepare_data():
    baseline = G.baseline
    paths = G.paths
    featmat = np.concatenate([baseline.features(path) for path in paths])
    returns = np.concatenate([path["returns"] for path in paths])
    G.par_nn_baseline_input_vals = (featmat, returns)


def worker_f(f_name, params, *args):
    G.baseline.set_param_values(params, trainable=True)
    return G.nn_baseline_opt_info[f_name](
        *(G.par_nn_baseline_input_vals + args))


def master_f(f_name):
    def f(params, *args):
        return parallel_sampler.master_collect_mean(
            worker_f, f_name, params, *args)
    return f


class ParNNBaseline(Baseline, LasagnePowered, Serializable):

    @autoargs.arg('hidden_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected hidden layers')
    @autoargs.arg('nonlinearity', type=str,
                  help='nonlinearity used for each hidden layer, can be one '
                       'of tanh, sigmoid')
    @autoargs.arg("optimizer", type=str,
                  help="Module path to the optimizer. It must support the "
                       "same interface as scipy.optimize.fmin_l_bfgs_b")
    @autoargs.arg("max_opt_itr", type=int,
                  help="Maximum number of batch optimization iterations.")
    def __init__(
            self,
            mdp,
            hidden_sizes=(32, 32),
            nonlinearity='lasagne.nonlinearities.tanh',
            optimizer='scipy.optimize.fmin_l_bfgs_b',
            max_opt_itr=20,
    ):
        super(ParNNBaseline, self).__init__(mdp)
        Serializable.__init__(
            self, mdp, hidden_sizes, nonlinearity, optimizer, max_opt_itr)

        self._optimizer = locate(optimizer)
        self._max_opt_itr = max_opt_itr

        if isinstance(nonlinearity, str):
            nonlinearity = locate(nonlinearity)
        input_var = TT.matrix('input')
        l_input = L.InputLayer(shape=(None, self._feature_size(mdp)),
                               input_var=input_var)
        l_hidden = l_input
        for idx, hidden_size in enumerate(hidden_sizes):
            l_hidden = L.DenseLayer(
                l_hidden,
                num_units=hidden_size,
                nonlinearity=nonlinearity,
                W=lasagne.init.Normal(0.1),
                name="h%d" % idx)
        v_layer = L.DenseLayer(
            l_hidden,
            num_units=1,
            nonlinearity=None,
            W=lasagne.init.Normal(0.01),
            name="value")

        v_var = L.get_output(v_layer)
        LasagnePowered.__init__(self, [v_layer])

        self._f_value = compile_function([input_var], [v_var])
        self._opt_initialized = False
        self.v_var = v_var
        self.input_var = input_var

    def _feature_size(self, mdp):
        obs_dim = mdp.observation_shape[0]
        return obs_dim*2 + 3

    def features(self, path):
        o = np.clip(path["observations"], -10, 10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1)/100.0
        return np.concatenate([o, o**2, al, al**2, al**3], axis=1)

    @property
    @overrides
    def algorithm_parallelized(self):
        return True

    @overrides
    def fit(self):
        if not self._opt_initialized:
            logger.log("initializing worker baseline optimization")
            parallel_sampler.run_map(worker_init_opt)
            logger.log("initialized")
            self._opt_initialized = True

        parallel_sampler.run_map(worker_prepare_data)

        cur_params = self.get_param_values(trainable=True)

        def evaluate_cost(penalty):
            def evaluate(params):
                val = master_f("loss")(params)
                return val.astype(np.float64)
            return evaluate

        def evaluate_grad(penalty):
            def evaluate(params):
                grad = master_f("grads")(params)
                flattened_grad = flatten_tensors(map(np.asarray, grad))
                return flattened_grad.astype(np.float64)
            return evaluate

        loss_before = evaluate_cost(0)(cur_params)
        logger.record_tabular('vf_LossBefore', loss_before)

        opt_params, _, _ = self._optimizer(
            func=evaluate_cost(0), x0=cur_params,
            fprime=evaluate_grad(0),
            maxiter=self._max_opt_itr
        )

        self.set_param_values(opt_params, trainable=True)

        loss_after = evaluate_cost(0)(opt_params)
        logger.record_tabular('vf_LossAfter', loss_after)
        logger.record_tabular('vf_dLoss', loss_before - loss_after)

    @overrides
    def predict(self, path):
        return self._f_value(self.features(path))

    @overrides
    def get_param_values(self, **tags):
        return LasagnePowered.get_param_values(self, **tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        return LasagnePowered.set_param_values(self, flattened_params, **tags)
