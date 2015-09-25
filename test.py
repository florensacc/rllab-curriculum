from policy import DiscreteNNPolicy
from algo.trpo import TRPO
from mdp.atari_mdp import AtariMDP, OBS_RAM
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne

class TestPolicy(DiscreteNNPolicy):

    def new_network_outputs(self, state_shape, action_dims, input_var):
        l_input = L.InputLayer(shape=(None, state_shape[0]), input_var=input_var)
        l_hidden_1 = L.DenseLayer(l_input, 20, nonlinearity=NL.tanh, W=lasagne.init.Normal(0.01))
        output_layers = [L.DenseLayer(l_hidden_1, Da, nonlinearity=NL.softmax) for Da in action_dims]
        return output_layers

class EarlyStopMDP(AtariMDP):

    def step(self, states, action_indices, repeat=1):
        next_states, obs, rewards, dones = super(EarlyStopMDP, self).step(states, action_indices, repeat)
        for idx, reward in enumerate(rewards):
            if reward != 0:
                dones[idx] = True
        return next_states, obs, rewards, dones

mdp = EarlyStopMDP(rom_path="vendor/atari_roms/pong.bin", obs_type=OBS_RAM)
#policy = TestPolicy((128,), [4, 3])
#policy.set_parameters(policy.get_parameters())

trpo = TRPO(samples_per_itr=100000)
trpo.train(TestPolicy, mdp)
