import cgtcompat as theano
import cgtcompat.tensor as T#theano.tensor as T
import lasagne.layers as L
import numpy as np
from misc.tensor_utils import flatten_tensors, unflatten_tensors
from .base import ContinuousPolicy

def normal_pdf(x, mean, log_std):
    return T.exp(-T.square((x - mean) / std) / 2) / ((2*np.pi)**0.5 * std)

class ContinuousNNPolicy(ContinuousPolicy):

    def __init__(self, *args, **kwargs):
        super(ContinuousNNPolicy, self).__init__(*args, **kwargs)
        mean_layer, log_std_layer = self.new_network_outputs(
            self.observation_shape,
            self.n_actions,
            self.input_var
            )
        action_var = T.matrix("actions")
        mean_var = L.get_output(mean_layer)
        #log_std_var = theano.shared(np.zeros((1, self.n_actions)))#, broadcastable=[True, False])
        #self.log_std_var = log_std_var
        log_std_var = L.get_output(log_std_layer)
        self.pdist_var = T.concatenate([mean_var, log_std_var], axis=1)
        self.mean_log_std_func = theano.function([self.input_var], [mean_var, log_std_var], allow_input_downcast=True)
        self.params = L.get_all_params(
            L.concat([mean_layer, log_std_layer]),
            trainable=True
        )
        self.param_shapes = map(
            lambda x: theano.compat.get_value(x, borrow=True).shape,
            self.params
        )
        self.param_dtypes = map(
            lambda x: theano.compat.get_value(x, borrow=True).dtype,
            self.params
        )

    def compute_action_mean_log_std(self, states):
        return self.mean_log_std_func(states)

    def compute_action_probs(self, states, actions):
        return self.probs_func(states, actions)

    def get_param_values(self):
        return flatten_tensors(map(
            lambda x: theano.compat.get_value(x, borrow=True), self.params
        ))

    def set_param_values(self, flattened_params):
        param_values = unflatten_tensors(flattened_params, self.param_shapes)
        for param, dtype, value in zip(
                self.params,
                self.param_dtypes,
                param_values
                ):
            theano.compat.set_value(param, value.astype(dtype))

    # new_network_outputs should return two Lasagne layers, one for the action mean and one for the action log standard deviations
    def new_network_outputs(self, observation_shape, n_actions, input_var):
        raise NotImplementedError
    
    def split_pdist(self, pdist):
        mean = pdist[:, :self.n_actions]
        log_std = pdist[:, self.n_actions:]
        return mean, log_std
