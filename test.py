#!/usr/bin/python
import os
from policy import DiscreteNNPolicy
from algo import UTRPO, UTRPO_VTS
from mdp import MDP, AtariMDP
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import numpy as np
import inspect
from misc.console import tweak


class RAMPolicy(DiscreteNNPolicy):

    def new_network_outputs(self, observation_shape, action_dims, input_var):#, hidden_units=[256,128]):
        l_input = L.InputLayer(shape=(None, observation_shape[0]), input_var=input_var)
        l_hidden_1 = L.DenseLayer(l_input, num_units=256, nonlinearity=NL.tanh, W=lasagne.init.Normal(0.01), name="h1")
        l_hidden_2 = L.DenseLayer(l_hidden_1, num_units=128, nonlinearity=NL.tanh, W=lasagne.init.Normal(0.01), name="h2")
        output_layers = [L.DenseLayer(l_hidden_2, num_units=Da, nonlinearity=NL.softmax, name="output_%d" % idx) for idx, Da in enumerate(action_dims)]
        return output_layers


if __name__ == '__main__':
    mdp = tweak(AtariMDP, 'mdp')
    gen_mdp = lambda: mdp(rom_path="vendor/atari_roms/seaquest.bin", obs_type='ram')
    trpo = tweak(UTRPO_VTS, 'algo')(max_samples_per_itr=100000, exp_name='utrpo_vts_seaquest_4_16_64', time_scales=[4,16,64])
    #trpo = tweak(UTRPO_VTS, 'algo')(max_samples_per_itr=100000, exp_name='utrpo_vts_seaquest_4_16_64', time_scales=[4,16,64])
    #trpo = tweak(UTRPO_VTS, 'algo')(max_samples_per_itr=100000, exp_name='utrpo_vts_seaquest_64', time_scales=[64])
    trpo.train(gen_mdp=gen_mdp, gen_policy=RAMPolicy)
