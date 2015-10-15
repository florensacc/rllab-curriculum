# encoding=utf-8
#!/usr/bin/python
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
from simple_nn_policy import SimpleNNPolicy


class RAMPolicy(DiscreteNNPolicy):

    def new_network_outputs(self, observation_shape, action_dims, input_var):#, hidden_units=[256,128]):
        l_input = L.InputLayer(shape=(None, observation_shape[0]), input_var=input_var)
        l_hidden_1 = L.DenseLayer(l_input, num_units=256, nonlinearity=NL.tanh, W=lasagne.init.Normal(0.01), name="h1")
        l_hidden_2 = L.DenseLayer(l_hidden_1, num_units=128, nonlinearity=NL.tanh, W=lasagne.init.Normal(0.01), name="h2")
        output_layers = [L.DenseLayer(l_hidden_2, num_units=Da, nonlinearity=NL.softmax, name="output_%d" % idx) for idx, Da in enumerate(action_dims)]
        return output_layers

def gen_policy(mdp):
    return SimpleNNPolicy(mdp, hidden_sizes=[32, 32])#, deterministic=True)

if __name__ == '__main__':
    gen_mdp = HopperMDP
    #gen_policy = genSimpleNNPolicy#lambda mdp: 
    algo = UTRPOCont(max_samples_per_itr=1000, discount=0.98)#max_samples_per_itr=10, n_parallel=1)
    algo.train(gen_mdp=gen_mdp, gen_policy=gen_policy)
