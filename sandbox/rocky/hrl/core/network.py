from __future__ import print_function
from __future__ import absolute_import
import lasagne.layers as L
import lasagne.init as LI
import itertools
import numpy as np


class MergeMLP(object):
    """
    A more general version of the usual multi-layer perceptron. It first split the input layer into multiple
    branches, specified by the `branch_dim` parameter, passing each one of them into several layers,
    and then concatenate them together followed by some number of layers after that.
    """

    def __init__(self, input_shape, branch_dims, output_dim, branch_hidden_sizes, joint_hidden_sizes,
                 hidden_nonlinearity, output_nonlinearity, hidden_W_init=LI.GlorotUniform(),
                 hidden_b_init=LI.Constant(0.), output_W_init=LI.GlorotUniform(), output_b_init=LI.Constant(0.),
                 name=None, input_var=None):

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        input_dim = np.prod(input_shape)
        assert input_dim == np.sum(branch_dims)
        l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var)
        if len(input_shape) > 1:
            l_reshaped_in = L.ReshapeLayer(l_in, (input_dim,))
        else:
            l_reshaped_in = l_in

        l_branch_hids = []

        branch_offset = 0

        for branch_idx, branch_dim, hidden_sizes in zip(itertools.count(), branch_dims, branch_hidden_sizes):
            l_hid = L.SliceLayer(l_reshaped_in, slice(branch_offset, branch_offset+branch_dim), axis=-1)
            for idx, hidden_size in enumerate(hidden_sizes):
                l_hid = L.DenseLayer(
                    l_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="%sbranch_%d_hidden_%d" % (prefix, branch_idx, idx),
                    W=hidden_W_init,
                    b=hidden_b_init,
                )
            branch_offset += branch_dim
            l_branch_hids.append(l_hid)

        l_joint_hid = L.concat(l_branch_hids)

        for idx, hidden_size in enumerate(joint_hidden_sizes):
            l_joint_hid = L.DenseLayer(
                l_joint_hid,
                num_units=hidden_size,
                nonlinearity=hidden_nonlinearity,
                name="%sjoint_hidden_%d" % (prefix, idx),
                W=hidden_W_init,
                b=hidden_b_init,
            )

        l_out = L.DenseLayer(
            l_joint_hid,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
            name="%soutput" % (prefix,),
            W=output_W_init,
            b=output_b_init,
        )
        # self._layers.append(l_out)
        self._l_in = l_in
        self._l_out = l_out
        self._input_var = l_in.input_var
        self._output = L.get_output(l_out)

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._input_var

    @property
    def output(self):
        return self._output
