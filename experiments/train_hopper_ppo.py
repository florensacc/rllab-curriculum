import os
os.environ['CGT_COMPAT_MODE'] = 'theano'
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

class HopperValueFunction(object):

    def __init__(self):
        self.coeffs = None

    def get_param_values(self):
        return self.coeffs

    def set_param_values(self, val):
        self.coeffs = val

    def _features(self, path):
        o = np.clip(path["observations"], -10,10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1,1)/100.0
        return np.concatenate([o, o**2, al, al**2, al**3, np.ones((l,1))], axis=1)

    def fit(self, paths):
        return
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        self.coeffs = np.linalg.lstsq(featmat, returns)[0]

    def predict(self, path):
        if self.coeffs is None:
            return np.zeros(len(path["rewards"]))
        #import ipdb; ipdb.set_trace()
        return self._features(path).dot(self.coeffs)

def gen_policy(mdp):
    policy = SimpleNNPolicy(mdp, hidden_sizes=[32, 32])
    #data = np.load('data/hopper/itr_190.npz')
    #data = np.load('itr_184_20151018174527.npz')
    #policy.set_param_values(data['opt_policy_params'])
    return policy

def gen_vf():
    vf = HopperValueFunction()
    #data = np.load('data/hopper/itr_190.npz')
    #vf.set_param_values(data['vf_params'])
    return vf

if __name__ == '__main__':
    gen_mdp = HopperMDP
    algo = PPO(exp_name='hopper', max_samples_per_itr=100000, discount=0.98, n_parallel=8)
    algo.train(gen_mdp=gen_mdp, gen_policy=gen_policy, gen_vf=gen_vf)
