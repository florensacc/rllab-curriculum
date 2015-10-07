import lasagne.layers as L
import theano
import theano.tensor as T
from misc.tensor_utils import flatten_tensors, unflatten_tensors

class DiscreteNNQFunc(object):

    def __init__(self, observation_shape, action_dims, input_var):
        self.input_var = input_var
        self.observation_shape = observation_shape
        self.action_dims = action_dims
        self.network_outputs = self.new_network_outputs(
            observation_shape,
            action_dims,
            self.input_var
            )
        self.q_var = L.get_output(self.network_outputs[0])#, self.network_outputs)[0]
        self.q_func = theano.function(
            [self.input_var],
            self.q_var,#T.concatenate(self.q_vars, axis=1),
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

    def compute_q_val(self, states):
        return self.q_func(states)

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

    # new_network_outputs should return a list of Lasagne layers, each of which
    # outputs a tensor of normalized action probabilities
    def new_network_outputs(self, observation_shape, action_dims, input_var):
        raise NotImplementedError
