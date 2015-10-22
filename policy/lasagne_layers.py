import lasagne.layers as L
import lasagne
import cgtcompat.tensor as T

class ParamLayer(L.Layer):
    def __init__(self, incoming, num_units, param=lasagne.init.Constant(0.), trainable=True, **kwargs):
        super(ParamLayer, self).__init__(incoming, **kwargs)
        self.num_units = num_units
        self.param = self.add_param(param, (1, num_units), name="param", trainable=trainable)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        return T.tile(self.param, (input.shape[0], 1))#input, self.param)


class OpLayer(L.Layer):
    def __init__(self, incoming, op, **kwargs):
        super(OpLayer, self).__init__(incoming, **kwargs)
        self.op = op

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return self.op(input)
