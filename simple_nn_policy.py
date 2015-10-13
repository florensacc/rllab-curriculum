# encoding=utf-8

import os
from policy import DiscreteNNPolicy, ContinuousNNPolicy
from algo import UTRPO, UTRPO_VTS, UTRPOCont
from mdp import MDP, AtariMDP, HopperMDP, CartpoleMDP
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import numpy as np
import inspect
from misc.console import tweak
from functools import partial
#import cgtcompat as theano
import cgtcompat.tensor as T
from algo import CEM

def normal_pdf(x, mean, std):
    return T.exp(-T.square((x - mean) / std) / 2) / ((2*np.pi)**0.5 * std)

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




class SimpleNNPolicy(ContinuousNNPolicy):

    def __init__(self, mdp, hidden_sizes=[32,32], nonlinearity=NL.tanh, deterministic=False):
        self.input_var = T.matrix('input')
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.deterministic = deterministic
        super(SimpleNNPolicy, self).__init__(self.input_var, mdp)

    def kl(self, old_pdep_var, new_pdep_var):
        old_mean, old_std = old_pdep_var
        new_mean, new_std = new_pdep_var
        # mean: (N*A)
        # std: (N*A)
        # formula:
        # { (μ₁ - μ₂)² + σ₁² - σ₂² } / (2σ₂²) + ln(σ₂/σ₁)
        return (T.square(old_mean - new_mean) + T.square(old_std) - T.square(new_std)) / (2*T.square(new_std)) + T.log(new_std / old_std)

    def compute_entropy(self, pdep):
        mean, std = pdep
        return np.mean(np.sum(np.log(std*np.sqrt(2*np.pi*np.e)), axis=1))


    def likelihood(self, pdep_vars, action_var):
        mean, std = pdep_vars
        return T.prod(normal_pdf(action_var, mean, std), axis=1)
    
    def new_network_outputs(self, observation_shape, n_actions, input_var):
        l_input = L.InputLayer(shape=(None, observation_shape[0]), input_var=input_var)
        l_hidden = l_input
        for idx, hidden_size in enumerate(self.hidden_sizes):
            l_hidden = L.DenseLayer(l_hidden, num_units=hidden_size, nonlinearity=self.nonlinearity, W=lasagne.init.Normal(0.1), name="h%d" % idx)
        mean_layer = L.DenseLayer(l_hidden, num_units=n_actions, nonlinearity=None, W=lasagne.init.Normal(0.01), name="output_mean")
        if self.deterministic:
            std_layer = ParamLayer(l_input, num_units=n_actions, param=lasagne.init.Constant(0.), trainable=False, name="output_std")
        else:
            std_layer = OpLayer(ParamLayer(l_input, num_units=n_actions), op=T.exp, name="output_std")
        return mean_layer, std_layer
