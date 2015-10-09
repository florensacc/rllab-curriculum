import theano
import theano.tensor as T
import lasagne.layers as L
import numpy as np
from misc.tensor_utils import flatten_tensors, unflatten_tensors
from .base import DiscretePolicy


class DiscreteNNPolicy(DiscretePolicy):

    def __init__(self, *args, **kwargs):
        super(DiscreteNNPolicy, self).__init__(*args, **kwargs)
        self.network_outputs = self.new_network_outputs(
            self.observation_shape,
            self.action_dims,
            self.input_var
            )
        self.probs_vars = map(L.get_output, self.network_outputs)
        self.probs_func = theano.function(
            [self.input_var],
            T.concatenate(self.probs_vars, axis=1),
            allow_input_downcast=True
        )
        self.params = L.get_all_params(
            L.concat(self.network_outputs),
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

    def compute_action_probs(self, states):
        action_probs = self.probs_func(states)
        indices = np.cumsum(self.action_dims)[:-1]
        return np.split(action_probs, indices, axis=1)

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

    def fisher_vector_product(self, probs_old, probs_new, eval_at):
        # For KL(p_old||p_new)
        return eval_at / (probs_old / (T.square(probs_new)))
        # For KL(p_new||p_old)
        # return eval_at / (probs_old / (probs_new))

    # new_network_outputs should return a list of Lasagne layers, each of which
    # outputs a tensor of normalized action probabilities
    def new_network_outputs(self, observation_shape, action_dims, input_var):
        raise NotImplementedError
