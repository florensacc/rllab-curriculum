import numpy as np
from misc.tensor_utils import high_res_normalize


def head(x):
    return x[0]


class DiscretePolicy(object):

    # observation_shape: Shape of observation
    # action_dims: A list of action dimensions. The actions are assumed to be
    # conditionally independent given the state (i.e. they're factored)
    def __init__(self, observation_shape, action_dims, input_var):
        self.input_var = input_var
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

    @property
    def params(self):
        raise NotImplementedError
