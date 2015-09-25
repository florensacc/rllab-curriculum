import numpy as np
import theano.tensor as T

def head(x):
    return x[0]

class DiscretePolicy(object):

    # state_shape: Shape of state
    # action_dims: A list of action dimensions. The actions are assumed to be
    # conditionally independent given the state (i.e. they're factored)
    def __init__(self, state_shape, action_dims, input_var):
        self.input_var = input_var
        self.state_shape = state_shape
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
    # The second item is a list of matrices, same as the result of compute_action_probs
    def get_actions(self, states):
        action_probs = self.compute_action_probs(states)
        action_indices = []
        for per_action_probs in action_probs:
            per_state_indices = []
            for per_state_probs in per_action_probs:
                a = np.random.choice(range(len(per_state_probs)), p=per_state_probs)
                per_state_indices.append(a)
            action_indices.append(per_state_indices)
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
