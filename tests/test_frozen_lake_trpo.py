from mdp import FrozenLakeMDP, ObsTransformer
from policy import DiscreteNNPolicy
from algo import TRPO
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import numpy as np

def process_state(state):
    Dx = 4
    Dy = 4
    s = np.zeros(Dx*Dy)
    s[state[0]*Dy+state[1]] = 1
    return s

class LinearPolicy(DiscreteNNPolicy):

    def new_network_outputs(self, observation_shape, action_dims, input_var):
        l_input = L.InputLayer(shape=(None, observation_shape[0]), input_var=input_var)
        l_hidden = L.DenseLayer(l_input, num_units=action_dims[0], nonlinearity=NL.softmax, W=lasagne.init.Normal(0.01), name="output")
        return [l_hidden]


desc = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
    ]

gen_mdp = lambda: ObsTransformer(FrozenLakeMDP(desc), process_state)
trpo = TRPO(max_samples_per_itr=10000)
trpo.train(gen_mdp, LinearPolicy)
