#!/usr/bin/python
import os
os.environ['CGT_COMPAT_MODE'] = 'cgt'
from policy import DiscreteNNPolicy, ContinuousNNPolicy
from algo import PPO
from mdp import MDP, AtariMDP, HopperMDP, CartpoleMDP
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import numpy as np
import inspect
from misc.console import tweak
from functools import partial
import cgtcompat.tensor as T
from algo import CEM
from simple_nn_policy import SimpleNNPolicy

def gen_policy(mdp):
    return SimpleNNPolicy(mdp, hidden_sizes=[32, 32])

if __name__ == '__main__':
    gen_mdp = HopperMDP
    algo = PPO(exp_name='hopper', max_samples_per_itr=100000, discount=0.98)
    algo.train(gen_mdp=gen_mdp, gen_policy=gen_policy)
