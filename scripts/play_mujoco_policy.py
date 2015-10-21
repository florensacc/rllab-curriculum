import os
os.environ['CGT_COMPAT_MODE'] = 'theano'
import numpy as np
from simple_nn_policy import SimpleNNPolicy, ParamLayer, OpLayer
from mdp import MDP, AtariMDP, HopperMDP, CartpoleMDP
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import cgtcompat as cgt
import cgtcompat.tensor as T

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
        self.log_std_vars = L.get_all_params(log_std_layer, trainable=True)
        return mean_layer, log_std_layer



def gen_policy(mdp):
    return SimpleNNPolicy(mdp, hidden_sizes=[32, 32])

print 'reading data'
data = np.load('itr_452.npz')
print 'read data'

params = data['cur_policy_params']
print params.shape
mdp = HopperMDP()
policy = gen_policy(mdp)
print policy.get_param_values().shape
#print policy.param_shapes
policy.set_param_values(params)
# zero out the variance
policy.log_std_vars[0].set_value(np.ones_like(policy.log_std_vars[0].get_value()) * -100)
#cur_params = policy.get_param_values()
mdp.demo_policy(policy)
