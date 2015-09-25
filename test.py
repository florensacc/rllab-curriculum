from policy import DiscreteNNPolicy
from algo.trpo import TRPO
from mdp.base import MDP
from mdp.atari_mdp import AtariMDP, OBS_RAM
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import numpy as np
import inspect


class TestPolicy(DiscreteNNPolicy):

    def new_network_outputs(self, state_shape, action_dims, input_var):
        l_input = L.InputLayer(shape=(None, state_shape[0]), input_var=input_var)
        #l_hidden_1 = L.DenseLayer(l_input, , nonlinearity=NL.tanh, W=lasagne.init.Normal(0.01))
        l_hidden_2 = L.DenseLayer(l_input, 20, nonlinearity=NL.tanh, W=lasagne.init.Normal(0.01))
        output_layers = [L.DenseLayer(l_hidden_2, Da, nonlinearity=NL.softmax) for Da in action_dims]
        return output_layers

class ProxyMDP(MDP):

    def __init__(self, base_mdp):
        self._base_mdp = base_mdp

    def sample_initial_states(self, n):
        return self._base_mdp.sample_initial_states(n)

    @property
    def action_set(self):
        return self._base_mdp.action_set

    @property
    def action_dims(self):
        return self._base_mdp.action_dims

    @property
    def observation_shape(self):
        return self._base_mdp.observation_shape

    def step(self, states, action_indices):
        return self._base_mdp.step(state, action_indices)

class ObsTransformer(ProxyMDP):

    def __init__(self, base_mdp, obs_transform):
        super(ObsTransformer, self).__init__(base_mdp)
        self._obs_transform = obs_transform

    def step(self, states, action_indices):
        next_states, obs, rewards, dones = self._base_mdp.step(states, action_indices)
        return next_states, map(self._obs_transform, obs), rewards, dones

class VariableTimeScaleMDP(ProxyMDP):

    def __init__(self, base_mdp, time_scales=[4,16,64]):
        super(VariableTimeScaleMDP, self).__init__(base_mdp)
        self._time_scales = time_scales
        self._has_repeat = 'repeat' in inspect.getargspec(self._base_mdp.step_single)

    @property
    def action_set(self):
        return self._base_mdp.action_set + range(len(self._time_scales))

    @property
    def action_dims(self):
        return self._base_mdp.action_dims + [len(self._time_scales)]

    def step(self, states, action_indices):
        next_states = []
        obs = []
        rewards = []
        dones = []
        for state, base_action, scale_action in zip(states, action_indices[:-1], action_indices[-1]):
            # sometimes, the mdp will support the repeat mechanism which saves the time required to obtain intermediate observations (ram / images)
            if self._has_repeat:
                next_state, ob, reward, done = self._base_mdp.step_single(state, base_action, repeat=self._time_scales[scale_action])
            else:
                reward = 0
                for _ in xrange(self._time_scales[scale_action]):
                    next_state, ob, step_reward, done = self._base_mdp.step_single(state, base_action)
                    reward += step_reward
                    if done:
                        break
            next_states.append(next_state)
            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
        return next_states, obs, rewards, dones

MAX_RAM = 255
def process_obs(ram_vector): # rescale the RAM vector to be between -1 and 1\n
    return (np.array(ram_vector) * 1.0 / MAX_RAM) * 2 - 1

def gen_mdp():
    return VariableTimeScaleMDP(
            ObsTransformer(
                AtariMDP(rom_path="vendor/atari_roms/pong.bin", obs_type=OBS_RAM, early_stop=True),
                process_obs
            ), time_scales=[4]
    )

trpo = TRPO(samples_per_itr=100000)
trpo.train(TestPolicy, gen_mdp)
