import theano
import theano.tensor as T
import lasagne.layers as L
import numpy as np
from misc.tensor_utils import flatten_tensors, unflatten_tensors
from .base import ContinuousPolicy

def normal_pdf(x, mean, std):
    return T.exp(-T.square((x - mean) / std) / 2) / ((2*np.pi)**0.5 * std)

class ContinuousNNPolicy(ContinuousPolicy):

    def __init__(self, *args, **kwargs):
        super(ContinuousNNPolicy, self).__init__(*args, **kwargs)
        mean_layer, std_layer = self.new_network_outputs(
            self.observation_shape,
            self.n_actions,
            self.input_var
            )
        action_var = T.matrix("actions")
        mean_var = L.get_output(mean_layer)
        std_var = L.get_output(std_layer)
        self.probs_var = normal_pdf(action_var, mean_var, std_var)
        self.mean_std_func = theano.function([self.input_var], [mean_var, std_var])
        self.probs_func = theano.function([self.input_var, action_var], self.probs_var)
        self.params = L.get_all_params(
            L.concat([mean_layer, std_layer]),
            trainable=True
        )
        self.param_shapes = map(
            lambda x: x.get_value(borrow=True).shape,
            self.params
        )
        self.param_dtypes = map(
            lambda x: x.get_value(borrow=True).dtype,
            self.params
        )

    def compute_action_mean_std(self, states):
        return self.mean_std_func(states)

    def compute_action_probs(self, states, actions):
        return self.probs_func(states, actions)

    def get_param_values(self):
        return flatten_tensors(map(
            lambda x: x.get_value(borrow=True), self.params
        ))

    def set_param_values(self, flattened_params):
        param_values = unflatten_tensors(flattened_params, self.param_shapes)
        for param, dtype, value in zip(
                self.params,
                self.param_dtypes,
                param_values
                ):
            param.set_value(value.astype(dtype))

    # new_network_outputs should return two Lasagne layers, one for the action mean and one for the action log standard deviations
    def new_network_outputs(self, observation_shape, n_actions, input_var):
        raise NotImplementedError
