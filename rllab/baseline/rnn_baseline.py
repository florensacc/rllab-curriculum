from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.lasagne_layers import OpLayer
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.ext import compile_function
from rllab.misc.tensor_utils import flatten_tensors, pad_tensor
from rllab.baseline.base import Baseline
from rllab.misc.overrides import overrides
import numpy as np
import theano
import theano.tensor as TT
import lasagne.layers as L
import lasagne.nonlinearities as NL
import scipy.optimize


class RNNBaseline(Baseline, LasagnePowered, Serializable):

    def __init__(self, mdp):
        super(RNNBaseline, self).__init__(mdp)
        Serializable.__init__(self, mdp)

        # baseline is a function from the history of observations to the
        # predicted value
        # bt = f(o1, o2, ..., ot)

        N_HIDDEN = 10
        GRAD_CLIP = 100

        l_obs = L.InputLayer(shape=(None, None, mdp.observation_shape[0]))
        l_prev_action = L.InputLayer(shape=(None, None, mdp.action_dim))
        l_in = L.ConcatLayer([l_obs, l_prev_action], axis=2)

        l_forward = L.LSTMLayer(
            l_in,
            num_units=N_HIDDEN,
            grad_clipping=GRAD_CLIP,
            nonlinearity=NL.tanh,
        )

        l_forward_reshaped = L.ReshapeLayer(l_forward, (-1, N_HIDDEN))

        l_raw_prediction = L.DenseLayer(
            l_forward_reshaped,
            num_units=1,
        )

        l_prediction = OpLayer(
            l_raw_prediction,
            op=lambda raw_prediction, input:
                TT.reshape(raw_prediction, (input.shape[0], input.shape[1])),
            shape_op=lambda _, input_shape: (input_shape[0], input_shape[1]),
            extras=[l_in],
        )

        Baseline.__init__(self, mdp)
        LasagnePowered.__init__(self, [l_prediction])
        Serializable.__init__(self, mdp)

        prediction_var = L.get_output(l_prediction)
        returns_var = TT.matrix('returns')
        # Account for the extra padded states
        valids_var = TT.matrix('valids')

        diff = prediction_var - returns_var

        loss = TT.sum(TT.square(diff * valids_var)) / TT.sum(valids_var)
        grad = theano.gradient.grad(loss, self.params)

        f_predict = compile_function(
            inputs=[l_obs.input_var, l_prev_action.input_var],
            outputs=prediction_var
        )
        f_loss = compile_function(
            inputs=[l_obs.input_var, l_prev_action.input_var, returns_var, valids_var],
            outputs=loss,
        )
        f_grad = compile_function(
            inputs=[l_obs.input_var, l_prev_action.input_var, returns_var, valids_var],
            outputs=grad
        )

        self._l_obs = l_obs
        self._l_prev_action = l_prev_action
        self._f_predict = f_predict
        self._f_loss = f_loss
        self._f_grad = f_grad
        self._trained = False

    def _prev_actions(self, path):
        return np.concatenate([
            np.zeros((1,) + path["actions"][0].shape),
            path["actions"][:-1]
        ])

    @overrides
    def fit(self, paths):
        max_path_length = max([len(path["returns"]) for path in paths])
        obs = [path["observations"] for path in paths]
        obs = [pad_tensor(ob, max_path_length, ob[0]) for ob in obs]
        prev_actions = map(self._prev_actions, paths)
        prev_actions = [pad_tensor(a, max_path_length, a[0]) for a in prev_actions]
        returns = [path["returns"] for path in paths]
        returns = [pad_tensor(r, max_path_length, 0) for r in returns]
        valids = [np.ones_like(path["returns"]) for path in paths]
        valids = [pad_tensor(v, max_path_length, 0) for v in valids]

        def evaluate(params):
            self.set_param_values(params)
            val = self._f_loss(obs, prev_actions, returns, valids)
            return val.astype(np.float64)

        def evaluate_grad(params):
            self.set_param_values(params)
            grad = self._f_grad(obs, prev_actions, returns, valids)
            flattened_grad = flatten_tensors(map(np.asarray, grad))
            return flattened_grad.astype(np.float64)

        cur_params = self.get_param_values()
        logger.log("Running optimization...")
        scipy.optimize.fmin_l_bfgs_b(
            func=evaluate, x0=cur_params, fprime=evaluate_grad, maxiter=10
        )

        self._trained = True

    @overrides
    def predict(self, path):
        if self._trained:
            obs = np.array([path["observations"]])
            prev_actions = np.array([self._prev_actions(path)])
            return self._f_predict(obs, prev_actions)
        else:
            return np.zeros_like(path["returns"])

    @overrides
    def get_param_values(self):
        return LasagnePowered.get_param_values(self)

    @overrides
    def set_param_values(self, flattened_params):
        return LasagnePowered.set_param_values(self, flattened_params)
