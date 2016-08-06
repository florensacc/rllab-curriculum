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
        return n_batch, new_length

    def get_output_for(self, all_obs_var, **kwargs):
        return all_obs_var[:, self.start_index:self.end_index]
