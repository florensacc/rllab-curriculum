import numpy as np
from misc.tensor_utils import high_res_normalize
import scipy.stats


def head(x):
    return x[0]

class DiscretePolicy(object):

    # observation_shape: Shape of observation
    # action_dims: A list of action dimensions. The actions are assumed to be
    # conditionally independent given the state (i.e. they're factored)
    def __init__(self, input_var, mdp=None, observation_shape=None, action_dims=None):
        self.input_var = input_var
        if mdp:
            self.observation_shape = mdp.observation_shape
            self.action_dims = mdp.action_dims
        else:
            self.observation_shape = observation_shape
            self.action_dims = action_dims

    # The return value is a list of matrices, each corresponding to the action
    # distributions of a single action, stacked horizontally for all states
    # i.e. [a1_probs, a2_probs, ..., an_probs]
    # where each ai_probs is of the form [ai_s1_probs, ai_s2_probs, ...]
    def compute_action_probs(self, states):
        raise NotImplementedError

    def compute_action_probs_single(self, state):
        return map(head, self.compute_action_probs([state]))

    # The return value is a pair. The first item is a list of integer-valued
    # vectors, each corresponding to the action index for a single action,
    # stacked for all states. i.e. [a1_indices, a2_indices, ..., an_indices]
    # where each ai_indices is of the form [ai_s1_idx, ai_s2_idx, ...]
    # The second item is a list of matrices, same as the result of
    # compute_action_probs
    def get_actions(self, states):
        action_probs = self.compute_action_probs(states)
        action_indices = [[] for _ in range(len(action_probs))]
        for idx, per_action_probs in enumerate(action_probs):
            for per_state_probs in per_action_probs:
                a = np.random.choice(range(len(per_state_probs)),
                                     p=high_res_normalize(per_state_probs))
                action_indices[idx].append(a)
        return action_indices, action_probs

    def get_actions_single(self, state):
        probs, indices = self.get_actions([state])
        return map(head, probs), map(head, indices)

    def get_param_values(self):
        raise NotImplementedError

    def set_param_values(self, flattened_parameters):
        raise NotImplementedError

class ContinuousPolicy(object):

    # observation_shape: Shape of observation
    # n_actions: Number of actions. They are expected to roughly lie in the range -3~3, and they are assumed
    # to be conditionally independent given the state
    def __init__(self, input_var, mdp=None, observation_shape=None, n_actions=None):
        self.input_var = input_var
        if mdp:
            self.observation_shape = mdp.observation_shape
            self.n_actions = mdp.n_actions
        else:
            self.observation_shape = observation_shape
            self.n_actions = n_actions

    def compute_action_mean_log_std(self, states):
        raise NotImplementedError

    def compute_action_probs(self, states, actions):
        raise NotImplementedError

    def compute_action_mean_log_std_single(self, state):
        return map(head, self.compute_action_mean_log_std([state]))

    def compute_action_probs_single(self, state):
        return map(head, self.compute_action_probs([state]))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    def get_actions(self, states):
        means, log_stds = self.compute_action_mean_log_std(states)
        # first get standard normal samples
        rnd = np.random.randn(*means.shape)#size=means.shape)
        pdeps = [means, log_stds]#scipy.stats.norm.pdf(rnd)
        # transform back to the true distribution
        actions = rnd * np.exp(log_stds) + means
        return actions, pdeps

    def get_action(self, state):
        actions, pdeps = self.get_actions([state])
        return head(actions), map(head, pdeps)

    def get_param_values(self):
        raise NotImplementedError

    def set_param_values(self, flattened_parameters):
        raise NotImplementedError
