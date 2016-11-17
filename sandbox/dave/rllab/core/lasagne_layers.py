# encoding: utf-8
import numpy as np
import lasagne.layers as L
import lasagne.nonlinearities as LN
import lasagne.init as LI
import theano.tensor as TT
import theano as T
from rllab.misc import ext
from rllab.core.lasagne_layers import OpLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from sandbox.dave.rllab.core. layers import DataLayer, ConvPoolLayer, DropoutLayer, FCLayer, SoftmaxLayer

class CropLayer(L.Layer):
    def __init__(self, l_incoming, start_index=None, end_index=None, name=None):
        super(CropLayer, self).__init__(l_incoming, name)
        self.start_index = start_index
        self.end_index = end_index

    def get_output_shape_for(self, input_shape):
        n_batch = input_shape[0]  # out of the obs_robot_var, the first dim is the batch size
        start = 0
        end = input_shape[1]
        if self.start_index:
            start = self.start_index
        if self.end_index:
            end = self.end_index
        new_length = end - start
        return n_batch, new_length  # this automatically creates a tuple

    def get_output_for(self, all_obs_var, **kwargs):
        return all_obs_var[:, self.start_index:self.end_index]


class ConstOutputLayer(L.Layer):
    def __init__(self, output_var=None, incoming=None, name=None, input_var=None, input_shape=None):
        super(ConstOutputLayer, self).__init__(incoming, name)
        self.output_var = output_var

    def get_output_shape_for(self, input_shape):
        n_batch = input_shape[0]  # the batch size
        single_output_shape = self.output_var.get_value().shape
        return (n_batch,) + single_output_shape  # this is supposed to create a tuple

    def get_output_for(self, all_obs_var, **kwargs):
        n_batch = all_obs_var.shape[0]
        out = TT.tile(self.output_var, (n_batch, 1))
        return out

class ElemwiseDiffLayer(L.MergeLayer):
    def get_output_shape_for(self, input_shapes):
        if any(shape != input_shapes[0] for shape in input_shapes):
            raise ValueError("Mismatch: not all input shapes are the same")
        return input_shapes[0]

    def get_output_for(self, inputs):
        output = None
        for input in inputs:
            if output is not None:
                output = output - input
            else:
                output = input
        return output

class FlattenLayer(L.Layer):
    """
    A layer that flattens its input. The leading ``outdim-1`` dimensions of
    the output will have the same shape as the input. The remaining dimensions
    are collapsed into the last dimension.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    outdim : int
        The number of dimensions in the output.
    See Also
    --------
    flatten  : Shortcut
    """
    def __init__(self, incoming, outdim=2, **kwargs):
        super(FlattenLayer, self).__init__(incoming, **kwargs)
        self.outdim = outdim

        if outdim < 1:
            raise ValueError('Dim must be >0, was %i', outdim)

    def get_output_shape_for(self, input_shape):
        to_flatten = input_shape[self.outdim - 1:]

        if any(s is None for s in to_flatten):
            flattened = None
        else:
            flattened = int(np.prod(to_flatten))

        return input_shape[:self.outdim - 1] + (flattened,)

    def get_output_for(self, input, **kwargs):
        return input.flatten(self.outdim)

flatten = FlattenLayer  # shortcut
