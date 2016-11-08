# encoding: utf-8

import lasagne.layers as L
import lasagne
import theano
import theano.tensor as TT
import numpy as np


class BilinearIntegrationLayer(L.MergeLayer):
    def __init__(self, incomings, name=None):  # incomings is a list (tuple) of 2 layers. The second is the "selector"
        super(BilinearIntegrationLayer, self).__init__(incomings, name)

    def get_output_shape_for(self, input_shapes):
        n_batch = input_shapes[0][0]  # out of the obs_robot_var, the first dim is the batch size
        robot_dim = input_shapes[0][1]
        selection_dim = input_shapes[1][1]
        return n_batch, robot_dim + selection_dim + robot_dim * selection_dim

    def get_output_for(self, inputs, **kwargs):
        obs_robot_var = inputs[0]
        selection_var = inputs[1]

        bilinear = TT.concatenate([obs_robot_var, selection_var,
                                   TT.flatten(obs_robot_var[:, :, np.newaxis] * selection_var[:, np.newaxis, :],
                                              outdim=2)]
                                  , axis=1)
        return bilinear


class SumProdLayer(L.MergeLayer):
    def __init__(self, incomings, name=None):  # prod is a numpy vector (or list) with the scalars to multiply
        super(SumProdLayer, self).__init__(incomings, name)  # each incoming layer before summing them, LAST IS COEFS!!
        # check if all input shapes are the same. See that the first dim might be NONE (for batch)
        coef_layer = incomings[-1]
        assert self.input_shapes[1:-1] == self.input_shapes[:-2]
        assert coef_layer.output_shape[0] == len(incomings) - 1 or coef_layer.output_shape[1] == len(incomings) - 1
        self.coef_layer = coef_layer

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]  # they are all supposed to be the same

    def get_output_for(self, inputs, **kwargs):
        coefs = inputs[-1]
        output = TT.zeros_like(inputs[0])
        for i, input_arr in enumerate(inputs[:-1]):
            output += input_arr * coefs[:, i].reshape((-1, 1))
        return output


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
        print('ConstOutputLayer is deprecated, use ParamLayer')
        super(ConstOutputLayer, self).__init__(incoming, name)
        self.output_var = output_var

    def get_output_shape_for(self, input_shape):
        # n_batch = input_shape[:-1]  # the batch size
        single_output_shape = self.output_var.get_value().shape
        # return (n_batch,) + single_output_shape  # this is supposed to create a tuple
        return input_shape[:-1] + single_output_shape

    def get_output_for(self, all_obs_var, **kwargs):
        # n_batch = all_obs_var.shape[:-1]
        # out = TT.tile(self.output_var, (n_batch, 1))
        # out = TT.tile(self.output_var, TT.concatenate([n_batch, [1]]))
        # return out
        ndim = all_obs_var.ndim
        reshaped_cnt = TT.reshape(self.output_var, (1,) * (ndim - 1) + self.output_var.get_value().shape)
        tile_arg = TT.concatenate([all_obs_var.shape[:-1], [1]])
        tiled = TT.tile(reshaped_cnt, tile_arg, ndim=ndim)
        return tiled

