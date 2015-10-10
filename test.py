#!/usr/bin/python
import os
from policy import DiscreteNNPolicy, ContinuousNNPolicy
from algo import UTRPO, UTRPO_VTS
from mdp import MDP, AtariMDP, HopperMDP, CartpoleMDP
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import numpy as np
import inspect
from misc.console import tweak
from functools import partial
import theano.tensor as T
from algo import CEM


class RAMPolicy(DiscreteNNPolicy):

    def new_network_outputs(self, observation_shape, action_dims, input_var):#, hidden_units=[256,128]):
        l_input = L.InputLayer(shape=(None, observation_shape[0]), input_var=input_var)
        l_hidden_1 = L.DenseLayer(l_input, num_units=256, nonlinearity=NL.tanh, W=lasagne.init.Normal(0.01), name="h1")
        l_hidden_2 = L.DenseLayer(l_hidden_1, num_units=128, nonlinearity=NL.tanh, W=lasagne.init.Normal(0.01), name="h2")
        output_layers = [L.DenseLayer(l_hidden_2, num_units=Da, nonlinearity=NL.softmax, name="output_%d" % idx) for idx, Da in enumerate(action_dims)]
        return output_layers

class ParamLayer(L.Layer):
    def __init__(self, incoming, num_units, param=lasagne.init.Constant(0.), trainable=True, **kwargs):
        super(ParamLayer, self).__init__(incoming, **kwargs)
        self.num_units = num_units
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.param = self.add_param(param, (1, num_units), name="param", trainable=trainable)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        return T.fill(input, self.param)

class OpLayer(L.Layer):
    def __init__(self, incoming, op, **kwargs):
        super(OpLayer, self).__init__(incoming, **kwargs)
        self.op = op

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return self.op(input)

class SimpleNNPolicy(ContinuousNNPolicy):

    def __init__(self, input_var, mdp, hidden_sizes=[32,32], nonlinearity=NL.tanh, deterministic=False):
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.deterministic = deterministic
        super(SimpleNNPolicy, self).__init__(input_var, mdp)
    
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

if __name__ == '__main__':
    gen_mdp = partial(HopperMDP)
    gen_policy = partial(SimpleNNPolicy, hidden_sizes=[32, 32], deterministic=True)
    algo = CEM()
    algo.train(gen_mdp=gen_mdp, gen_policy=gen_policy)
