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
from simple_nn_policy import SimpleNNPolicy, ParamLayer, OpLayer

np.random.seed(0)


class ModifiedNNPolicy(SimpleNNPolicy):
    def new_network_outputs(self, observation_shape, n_actions, input_var):
        l_input = L.InputLayer(shape=(None, observation_shape[0]), input_var=input_var)
        l_hidden = l_input
        for idx, hidden_size in enumerate(self.hidden_sizes):
            l_hidden = L.DenseLayer(l_hidden, num_units=hidden_size, nonlinearity=self.nonlinearity, W=lasagne.init.Normal(0.1), name="h%d" % idx)
        mean_layer = L.DenseLayer(l_hidden, num_units=n_actions, nonlinearity=None, W=lasagne.init.Normal(0.01), name="output_mean")

        # some output between 0 and 1
        base_std_layer = L.DenseLayer(l_hidden, num_units=n_actions, nonlinearity=NL.sigmoid, W=lasagne.init.Normal(0.01))#, name="output_mean")
        # trainable bias parameter
        std_layer = L.BiasLayer(base_std_layer, b=lasagne.init.Constant(1.))
        #addn_std_layer = ParamLayer(l_input, num_units=n_actions, param=lasagne.init.Constant(1.))#, name="output_log_std")
        #std_layer = L.ElemwiseSumLayer([l_base_std, l_addn_std])
        log_std_layer = OpLayer(std_layer, T.log)
        self.log_std_var = L.get_all_params(log_std_layer, trainable=True)[0]
        return mean_layer, log_std_layer

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
        #return
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
    algo = PPO(exp_name='hopper_10k', max_samples_per_itr=10000, discount=0.98, n_parallel=4, stepsize=0.0016)
    algo.train(gen_mdp=gen_mdp, gen_policy=gen_policy, gen_vf=gen_vf)
