from mdp import FrozenLakeMDP, ObsTransformer
from policy import DiscreteNNPolicy
from algo import UTRPO
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import numpy as np

def process_state(state):
    Dx = 8
    Dy = 8
    s = np.zeros(Dx*Dy)
    s[state[0]*Dy+state[1]] = 1
    return s

class LinearPolicy(DiscreteNNPolicy):

    def new_network_outputs(self, observation_shape, action_dims, input_var):
        l_input = L.InputLayer(shape=(None, observation_shape[0]), input_var=input_var)
        l_hidden = L.DenseLayer(l_input, num_units=action_dims[0], nonlinearity=NL.softmax, W=lasagne.init.Constant(0), name="output")
        return [l_hidden]

#class TabularPolicy(DiscretePolicy):



#desc = [
#    "SFFF",
#    "FHFH",
#    "FFFH",
#    "HFFG"
#    ]

desc = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG"
]


gen_mdp = lambda: ObsTransformer(FrozenLakeMDP(desc), process_state)
trpo = UTRPO(exp_name='frozen_lake_utrpo', n_itr=100, max_samples_per_itr=10000, stepsize=0.1, n_parallel=1, adapt_lambda=False, initial_lambda=0.1, max_opt_itr=20)
trpo.train(gen_mdp, LinearPolicy)
