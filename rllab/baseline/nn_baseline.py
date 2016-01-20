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
    @autoargs.arg('use_tr', type=bool,
                  help='whether use trust region or not')
    @autoargs.arg("optimizer", type=str,
                  help="Module path to the optimizer. It must support the "
                       "same interface as scipy.optimize.fmin_l_bfgs_b")
    @autoargs.arg("max_opt_itr", type=int,
                  help="Maximum number of batch optimization iterations.")
    @autoargs.arg("step_size", type=float,
                  help="Maximum change in mean KL per iteration.")
    @autoargs.arg("initial_penalty", type=float,
                  help="Initial value of the penalty coefficient.")
    @autoargs.arg("min_penalty", type=float,
                  help="Minimum value of penalty coefficient.")
    @autoargs.arg("max_penalty", type=float,
                  help="Maximum value of penalty coefficient.")
    @autoargs.arg("increase_penalty_factor", type=float,
                  help="How much the penalty should increase if kl divergence "
                       "exceeds the threshold on the first penalty "
                       "iteration.")
    @autoargs.arg("decrease_penalty_factor", type=float,
                  help="How much the penalty should decrease if kl divergence "
                       "is less than the threshold on the first penalty "
                       "iteration.")
    @autoargs.arg("max_penalty_itr", type=int,
                  help="Maximum number of penalty iterations.")
    @autoargs.arg("adapt_penalty", type=bool,
                  help="Whether to adjust penalty for each iteration.")
    def __init__(
            self,
            mdp,
            hidden_sizes=(32, 32),
            nonlinearity='lasagne.nonlinearities.tanh',
            use_tr=False,
            optimizer='scipy.optimize.fmin_l_bfgs_b',
            max_opt_itr=20,
            step_size=0.1,
            initial_penalty=1e-2,
            min_penalty=1e-6,
            max_penalty=1e6,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            max_penalty_itr=10,
            adapt_penalty=True,
    ):
        super(NNBaseline, self).__init__(mdp)
        Serializable.__init__(self, mdp, hidden_sizes, nonlinearity, use_tr, optimizer,
                              max_opt_itr, step_size, initial_penalty, min_penalty, max_penalty,
                              increase_penalty_factor, decrease_penalty_factor, max_penalty_itr,
                              adapt_penalty)

        self._max_penalty_itr = max_penalty_itr
        self._decrease_penalty_factor = decrease_penalty_factor
        self._increase_penalty_factor = increase_penalty_factor
        self._max_penalty = max_penalty
        self._min_penalty = min_penalty
        self._penalty = initial_penalty
        self._step_size = step_size
        self._use_tr = use_tr
        self._optimizer = locate(optimizer)
        self._max_opt_itr = max_opt_itr
        self._adapt_penalty = adapt_penalty

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

        if use_tr:
            old_v_var = TT.vector("old_values")
            penalty_var = TT.scalar('penalty')
            tr_loss = TT.mean(TT.square(v_var - old_v_var[:, np.newaxis]))
            penalized_obj = loss + penalty_var * tr_loss
            input_list += [old_v_var, penalty_var]
            output_list = [penalized_obj, loss, tr_loss]
        else:
            output_list = [loss]

        grads = theano.gradient.grad(loss, self.get_params(trainable=True))

        self._f_loss = compile_function(input_list, output_list)
        self._f_grads = compile_function(input_list, grads)


    def _feature_size(self, mdp):
        obs_dim = mdp.observation_shape[0]
        return obs_dim*2 + 3

    def _features(self, path):
        o = np.clip(path["observations"], -10,10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1,1)/100.0
        return np.concatenate([o, o**2, al, al**2, al**3], axis=1)

    @overrides
    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        input_vals = [featmat, returns]
        if self._use_tr:
            old_vs = np.concatenate([self.predict(path)[0] for path in paths]).reshape(-1)
            input_vals += [old_vs]
            var = np.mean(np.square(old_vs - returns))

        cur_params = self.get_param_values(trainable=True)

        def evaluate_cost(penalty):
            def evaluate(params):
                self.set_param_values(params, trainable=True)
                if self._use_tr:
                    inputs_with_penalty = input_vals + [penalty]
                    val, _, _ = self._f_loss(*inputs_with_penalty)
                else:
                    val, = self._f_loss(*input_vals)
                return val.astype(np.float64)
            return evaluate

        def evaluate_grad(penalty):
            def evaluate(params):
                self.set_param_values(params, trainable=True)
                if self._use_tr:
                    grad = self._f_grads(*(input_vals + [penalty]))
                else:
                    grad = self._f_grads(*input_vals)
                flattened_grad = flatten_tensors(map(np.asarray, grad))
                return flattened_grad.astype(np.float64)
            return evaluate

        loss_before = evaluate_cost(0)(cur_params)
        logger.record_tabular('vf_LossBefore', loss_before)

        if self._use_tr:
            try_penalty = np.clip(self._penalty, self._min_penalty, self._max_penalty)

            # search for the best penalty parameter
            penalty_scale_factor = None
            opt_params = None
            max_penalty_itr = self._max_penalty_itr
            mean_kl = None
            search_succeeded = False
            for penalty_itr in range(max_penalty_itr):
                logger.log('vf trying penalty=%.3f...' % try_penalty)
                self._optimizer(
                    func=evaluate_cost(try_penalty), x0=cur_params,
                    fprime=evaluate_grad(try_penalty),
                    maxiter=self._max_opt_itr
                )
                _, try_loss, try_tr_loss = self._f_loss(*(input_vals + [try_penalty]))
                try_mean_kl = try_tr_loss / 2. / var
                logger.log('vf penalty %f => loss %f, mean kl %f' %
                           (try_penalty, try_loss, try_mean_kl))
                if try_mean_kl < self._step_size or \
                        (penalty_itr == max_penalty_itr - 1 and
                                 opt_params is None):
                    opt_params = self.get_param_values(trainable=True)
                    penalty = try_penalty
                    mean_kl = try_mean_kl

                if not self._adapt_penalty:
                    break

                # decide scale factor on the first iteration
                if penalty_scale_factor is None or np.isnan(try_mean_kl):
                    if try_mean_kl > self._step_size or np.isnan(try_mean_kl):
                        # need to increase penalty
                        penalty_scale_factor = self._increase_penalty_factor
                    else:
                        # can shrink penalty
                        penalty_scale_factor = self._decrease_penalty_factor
                else:
                    if penalty_scale_factor > 1 and \
                                    try_mean_kl <= self._step_size:
                        search_succeeded = True
                        break
                    elif penalty_scale_factor < 1 and \
                                    try_mean_kl >= self._step_size:
                        search_succeeded = True
                        break
                try_penalty *= penalty_scale_factor
                if try_penalty < self._min_penalty or \
                                try_penalty > self._max_penalty:
                    try_penalty = np.clip(
                        try_penalty, self._min_penalty, self._max_penalty)
                    opt_params = self.get_param_values(trainable=True)
                    penalty = try_penalty
                    mean_kl = try_mean_kl
                    break

            self._penalty = penalty
            logger.record_tabular('vf_MeanKL', mean_kl)
            self.set_param_values(opt_params, trainable=True)
        else:
            self._optimizer(
                func=evaluate_cost(0), x0=cur_params,
                fprime=evaluate_grad(0),
                maxiter=self._max_opt_itr
            )
            opt_params = self.get_param_values(trainable=True)
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
