import os
os.environ['CGT_COMPAT_MODE'] = 'theano'
import numpy as np
from simple_nn_policy import SimpleNNPolicy, ParamLayer, OpLayer
from mdp import MDP, AtariMDP, HopperMDP, CartpoleMDP

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



def gen_policy(mdp):
    return ModifiedNNPolicy(mdp, hidden_sizes=[32, 32])

print 'reading data'
data = np.load('data/hopper_per_state_std_30k/itr_48.npz')
print 'read data'

params = data['cur_policy_params']
mdp = HopperMDP()
policy = gen_policy(mdp)
print policy.param_shapes
#policy.set_param_values(params)
#cur_params = policy.get_param_values()
#import ipdb; ipdb.set_trace()
#mdp.demo_policy(policy)
