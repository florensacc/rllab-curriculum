#!/usr/bin/python
import os
os.environ['CGT_COMPAT_MODE'] = 'theano'
from qfunc import DiscreteNNQFunc
from algo import DQN
from mdp import MDP, AtariMDP, FrozenLakeMDP, ObsTransformer
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import numpy as np
import inspect
from misc.console import tweak
from functools import partial


class TableQFunc(DiscreteNNQFunc):

    def new_network_outputs(self, observation_shape, n_actions, input_var):
        l_input = L.InputLayer(shape=(None, observation_shape[0]), input_var=input_var)
        output_layers = [L.DenseLayer(l_input, num_units=n_actions, nonlinearity=None, name="output")]
        return output_layers

class RAMQFunc(DiscreteNNQFunc):

    def new_network_output(self, observation_shape, n_actions, input_var):#, hidden_units=[256,128]):
        l_input = L.InputLayer(shape=(None, observation_shape[0]), input_var=input_var)
        l_hidden_1 = L.DenseLayer(l_input, num_units=256, nonlinearity=NL.tanh, W=lasagne.init.Normal(0.01), name="h1")
        l_hidden_2 = L.DenseLayer(l_hidden_1, num_units=128, nonlinearity=NL.tanh, W=lasagne.init.Normal(0.01), name="h2")
        output_layer = L.DenseLayer(l_hidden_2, num_units=n_actions, nonlinearity=None, name="output")
        return output_layer

def process_state(state):
    Dx = 4
    Dy = 4
    s = np.zeros(Dx*Dy)
    s[state[0]*Dy+state[1]] = 1
    return s

desc = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
    ]



if __name__ == '__main__':
    #mdp = tweak(FrozenLakeMDP, 'mdp')
    #gen_mdp = partial(AtariMDP, FrozenLakeMDP#lambda: FrozenLakeMDP lambda: mdp(rom_path="vendor/atari_roms/pong.bin", obs_type='ram')
    gen_mdp = partial(AtariMDP, rom_path="vendor/atari_roms/pong.bin", obs_type='ram')
    #gen_mdp = lambda: ObsTransformer(FrozenLakeMDP(desc), process_state)
    #gen_mdp = partial(FrozenLakeMDP, desc = [
    #    "SFFF",
    #    "FHFH",
    #    "FFFH",
    #    "HFFG"
    #    ])

    dqn = DQN()#, 'algo')()#(max_samples_per_itr=10000, exp_name='utrpo_seaquest_4')#, time_scales=[4])
    #trpo = tweak(UTRPO_VTS, 'algo')(max_samples_per_itr=100000, exp_name='utrpo_vts_seaquest_4_16_64', time_scales=[4,16,64])
    #trpo = tweak(UTRPO_VTS, 'algo')(max_samples_per_itr=100000, exp_name='utrpo_vts_seaquest_64', time_scales=[64])
    dqn.train(gen_mdp=gen_mdp, gen_q_func=RAMQFunc)
