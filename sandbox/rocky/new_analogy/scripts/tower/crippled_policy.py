import numpy as np

import gpr.policy


class CrippledPolicy(gpr.policy.Policy):
    """
    Cripple the action of the wrapped policy so that it does not change rapidly compared to the change in the
    observation. Note however that this indicates that the resulting policy is not strictly memoryless
    """

    def __init__(self, wrapped_policy, max_ratio=1.):
        self.wrapped_policy = wrapped_policy
        self.last_obs = None
        self.last_action = None
        self.max_ratio = max_ratio

    def get_action(self, obs):
        candidate_action = self.wrapped_policy.get_action(obs)
        if self.last_obs is not None:
            flat_obs = np.concatenate(obs)
            flat_action = candidate_action.flatten()
            obs_diff = np.linalg.norm(flat_obs - self.last_obs)
            act_diff = np.linalg.norm(flat_action - self.last_action)
            if act_diff / (obs_diff + 1e-8) > self.max_ratio:
                diff = flat_action - self.last_action
                diff = diff / np.linalg.norm(diff) * self.max_ratio * obs_diff
                final_action = self.last_action + diff
            else:
                final_action = candidate_action
        else:
            final_action = candidate_action

        self.last_obs = np.concatenate(obs)
        self.last_action = final_action
        return final_action

    def reset(self):
        self.wrapped_policy.reset()
