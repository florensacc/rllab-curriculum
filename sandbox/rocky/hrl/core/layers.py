


import lasagne.layers as L
import theano.tensor as TT


class GateLayer(L.MergeLayer):
    def __init__(self, l_gate, l_current, l_prev, **kwargs):
        super(GateLayer, self).__init__([l_gate, l_current, l_prev], **kwargs)
        self.l_gate = l_gate
        self.l_current = l_current
        self.l_prev = l_prev

    def get_output_shape_for(self, input_shapes):
        gate_shape, current_shape, prev_shape = input_shapes
        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = tuple(next((s for s in sizes if s is not None), None)
                             for sizes in zip(current_shape, prev_shape))

        def match(shape1, shape2):
            return (len(shape1) == len(shape2) and
                    all(s1 is None or s2 is None or s1 == s2
                        for s1, s2 in zip(shape1, shape2)))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in (current_shape, prev_shape)):
            raise ValueError("Mismatch: l_current shape should be compatible with l_prev shape")
        return output_shape

    def get_output_for(self, inputs, **kwargs):
        gate_var, current_var, prev_var = inputs
        gate_var = TT.tile(gate_var, [1] * (current_var.ndim - 1) + [current_var.shape[-1]])
        return gate_var * current_var + (1 - gate_var) * prev_var
