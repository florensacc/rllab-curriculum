# encoding=utf-8

import os
from policy import DiscreteNNPolicy, ContinuousNNPolicy
from mdp import MDP, AtariMDP, HopperMDP, CartpoleMDP
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import numpy as np
import inspect
from misc.console import tweak
from functools import partial
import cgtcompat as cgt
import cgtcompat.tensor as T
from algo import CEM

def normal_pdf(x, mean, log_std):
    return T.exp(-T.square((x - mean) / T.exp(log_std)) / 2) / ((2*np.pi)**0.5 * T.exp(log_std))

def log_normal_pdf(x, mean, log_std):
    normalized = (x - mean) / T.exp(log_std)
    return -0.5*T.square(normalized) - np.log((2*np.pi)**0.5) - log_std

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

    def __init__(self, mdp, hidden_sizes=[32,32], nonlinearity=NL.tanh):
        self.input_var = T.matrix('input')
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        super(SimpleNNPolicy, self).__init__(self.input_var, mdp)

    def kl(self, old_pdist_var, new_pdist_var):
        old_mean, old_log_std = self.split_pdist(old_pdist_var)
        new_mean, new_log_std = self.split_pdist(new_pdist_var)
        old_std = T.exp(old_log_std)
        new_std = T.exp(new_log_std)
        # mean: (N*A)
        # std: (N*A)
        # formula:
        # { (μ₁ - μ₂)² + σ₁² - σ₂² } / (2σ₂²) + ln(σ₂/σ₁)
        return T.sum((T.square(old_mean - new_mean) + T.square(old_std) - T.square(new_std)) / (2*T.square(new_std) + 1e-8) + new_log_std - old_log_std, axis=1)

    def likelihood_ratio(self, old_pdist_var, new_pdist_var, action_var):
        old_mean, old_log_std = self.split_pdist(old_pdist_var)
        new_mean, new_log_std = self.split_pdist(new_pdist_var)
        logli_new = log_normal_pdf(action_var, new_mean, new_log_std)
        logli_old = log_normal_pdf(action_var, old_mean, old_log_std)
        #return T.sum(T.exp(logli_new - logli_old), axis=1)
        return T.exp(T.sum(logli_new - logli_old, axis=1))

    def compute_entropy(self, pdist):
        mean, log_std = self.split_pdist(pdist)
        return np.mean(np.sum(log_std + np.log(np.sqrt(2*np.pi*np.e)), axis=1))

    def likelihood(self, pdist_var, action_var):
        mean, log_std = self.split_pdist(pdist_var)
        return T.prod(normal_pdf(action_var, mean, log_std), axis=1)
    
    def new_network_outputs(self, observation_shape, n_actions, input_var):
        l_input = L.InputLayer(shape=(None, observation_shape[0]), input_var=input_var)
        l_hidden = l_input
        for idx, hidden_size in enumerate(self.hidden_sizes):
            l_hidden = L.DenseLayer(l_hidden, num_units=hidden_size, nonlinearity=self.nonlinearity, W=lasagne.init.Normal(0.1), name="h%d" % idx)
        mean_layer = L.DenseLayer(l_hidden, num_units=n_actions, nonlinearity=None, W=lasagne.init.Normal(0.01), name="output_mean")
        log_std_layer = ParamLayer(l_input, num_units=n_actions, param=lasagne.init.Constant(0.), name="output_log_std")
        self.log_std_var = L.get_all_params(log_std_layer, trainable=True)[0]
        return mean_layer, log_std_layer

    def print_debug(self):
        print 'log std values: ', cgt.compat.get_value(self.log_std_var)
