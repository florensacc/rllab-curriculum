import os
os.environ['CGT_COMPAT_MODE'] = 'cgt'
import numpy as np
from simple_nn_policy import SimpleNNPolicy
from mdp import MDP, AtariMDP, HopperMDP, CartpoleMDP

def gen_policy(mdp):
    return SimpleNNPolicy(mdp, hidden_sizes=[32, 32])

data = np.load('data/hopper/itr_190.npz')

params = data['opt_policy_params']
mdp = HopperMDP()
policy = gen_policy(mdp)
policy.set_param_values(params)
cur_params = policy.get_param_values()
import ipdb; ipdb.set_trace()
mdp.demo_policy(policy)
