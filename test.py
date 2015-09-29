#!/usr/bin/python
import os
from policy import DiscreteNNPolicy
from algo.utrpo import UTRPO
from mdp.base import MDP
from mdp.atari_mdp import AtariMDP
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import numpy as np
import inspect
from misc.console import tweak


class TestPolicy(DiscreteNNPolicy):

    def new_network_outputs(self, state_shape, action_dims, input_var):
        l_input = L.InputLayer(shape=(None, state_shape[0]), input_var=input_var)
        l_hidden_1 = L.DenseLayer(l_input, 256, nonlinearity=NL.tanh, W=lasagne.init.Normal(0.01))
        l_hidden_2 = L.DenseLayer(l_hidden_1, 128, nonlinearity=NL.tanh, W=lasagne.init.Normal(0.01))
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

class VariableTimeScaleMDP(ProxyMDP):

    def __init__(self, base_mdp, time_scales=[4,16,64]):
        super(VariableTimeScaleMDP, self).__init__(base_mdp)
        self._time_scales = time_scales

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
        steps = []
        for state, base_action, scale_action in zip(states, action_indices[:-1], action_indices[-1]):
            # sometimes, the mdp will support the repeat mechanism which saves the time required to obtain intermediate observations (ram / images)
            if self._base_mdp.support_repeat:#self._has_repeat:
                next_state, ob, reward, done, step = self._base_mdp.step_single(state, base_action, repeat=self._time_scales[scale_action])
            else:
                reward = 0
                step = 0
                next_state = state
                for _ in xrange(self._time_scales[scale_action]):
                    next_state, ob, step_reward, done, per_step = self._base_mdp.step_single(next_state, base_action)
                    reward += step_reward
                    step += per_step
                    if done:
                        break
            # experiment with counter the effect
            #effective_step /= self._time_scales[scale_action]
            next_states.append(next_state)
            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
            #print effective_step
            steps.append(step)
        return next_states, obs, rewards, dones, steps

def gen_mdp():
    return tweak(AtariMDP)(rom_path="vendor/atari_roms/seaquest.bin", obs_type='ram')

if __name__ == '__main__':
    trpo = tweak(UTRPO)(max_samples_per_itr=100000)
    trpo.train(gen_mdp=gen_mdp, gen_policy=TestPolicy)
