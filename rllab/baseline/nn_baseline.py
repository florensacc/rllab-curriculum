from pydoc import locate

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
from rllab.misc.ext import compile_function
from rllab.misc.tensor_utils import flatten_tensors
from rllab.baseline.base import Baseline
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
import numpy as np
import theano
import theano.tensor as TT
import lasagne.layers as L
import lasagne

class NNBaseline(Baseline, LasagnePowered, Serializable):

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
        super(NNBaseline, self).__init__(mdp)
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

        new_v_var = TT.vector("new_values")
        loss = TT.mean(TT.square(v_var - new_v_var[:, np.newaxis]))
        input_list = [input_var, new_v_var]

        output_list = [loss]

        grads = theano.gradient.grad(loss, self.get_params(trainable=True))

        self._f_loss = compile_function(input_list, output_list)
        self._f_grads = compile_function(input_list, grads)


    def _feature_size(self, mdp):
        obs_dim = mdp.observation_shape[0]
        return obs_dim

    def _features(self, path):
        return path["observations"]

    @overrides
    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        input_vals = [featmat, returns]

        cur_params = self.get_param_values(trainable=True)

        def evaluate_cost(penalty):
            def evaluate(params):
                self.set_param_values(params, trainable=True)
                val, = self._f_loss(*input_vals)
                return val.astype(np.float64)
            return evaluate

        def evaluate_grad(penalty):
            def evaluate(params):
                self.set_param_values(params, trainable=True)
                grad = self._f_grads(*input_vals)
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
        return self._f_value(self._features(path))

    @overrides
    def get_param_values(self, **tags):
        return LasagnePowered.get_param_values(self, **tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        return LasagnePowered.set_param_values(self, flattened_params, **tags)
