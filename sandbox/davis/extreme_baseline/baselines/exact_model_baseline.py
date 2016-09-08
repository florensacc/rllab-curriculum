from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.special import discount_return

import numpy as np


class ExactModelBaseline(Baseline):
    def __init__(
            self,
            env_spec,
            lookahead=0,
            num_rollouts_per_state=10,
            discount=1,
            env=None,
            extreme=True,
            value_estimator=LinearFeatureBaseline,
            **value_estimator_kwargs
    ):
        self.env_spec = env_spec
        self.lookahead = lookahead
        self.num_rollouts_per_state = num_rollouts_per_state
        self.discount = discount
        self.env = env
        self.extreme = extreme
        self.value_estimator = value_estimator(env_spec, **value_estimator_kwargs)

    @overrides
    def get_param_values(self, **tags):
        return self.value_estimator.get_param_values(**tags)

    @overrides
    def set_param_values(self, val, **tags):
        self.value_estimator.set_param_values(val, **tags)

    @overrides
    def extreme(self):
        return True

    @overrides
    def fit(self, paths):
        self.value_estimator.fit(paths)

    @overrides
    def predict(self, path, policy):
        path_length, obs_dim = path['observations'].shape
        final_observations = np.zeros((self.num_rollouts_per_state, path_length, obs_dim))
        empirical_returns = np.zeros((self.num_rollouts_per_state, path_length))
        for t, env_state in enumerate(path['env_states']):
            noise = path['agent_infos']['noise'][t:]
            curr_obs = path['observations'][t]
            max_path_length = min(self.lookahead, path_length - t)
            for itr in range(self.num_rollouts_per_state):
                final_observation, return_so_far = self.partial_rollout_fixed_policy_noise(
                    policy, noise, curr_obs, env_state, max_path_length)
                final_observations[itr][t] = final_observation
                empirical_returns[itr][t] = return_so_far
        for itr, final in enumerate(final_observations):
            value_estimator_path = dict(
                observations=shift_with_zero_padding(final, self.lookahead))
            value_estimates = self.value_estimator.predict(value_estimator_path, policy)
            value_estimates = shift_with_zero_padding(value_estimates, -self.lookahead)
            empirical_returns[itr] += self.discount**self.lookahead * value_estimates
        return np.average(empirical_returns, axis=0)

    def partial_rollout_fixed_policy_noise(
            self, policy, noise, curr_obs, curr_state, max_path_length
            ):
        rewards = []
        obs = curr_obs
        path_length = 0
        for _ in self.env.set_state_tmp(curr_state):
            while path_length < max_path_length:
                if self.extreme and path_length > 0:  # Don't fix noise for first action
                    action, _ = policy.get_action_with_fixed_noise(obs, noise[path_length])
                else:
                    action, _ = policy.get_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                path_length += 1
                obs = next_obs
                if done:
                    break
            return_so_far = discount_return(np.array(rewards), self.discount)
        return obs, return_so_far

    # def partial_rollout_fixed_policy_noise_point_env_vectorized(
    #         self, policy, noise, curr_obss, curr_states, max_path_length
    #         ):
    #     rewards = []
    #     obss = curr_obss
    #     path_length = 0
    #     while path_length < max_path_length:
    #         actions, _ = policy.


def shift_with_zero_padding(arr, shift):
    """
    Shifts an array along the vertical axis (arr[i] gets shifted up or down for 3d,
    whole array gets shifted up or down for 2d), but fills in with zeros instead of rolling
    values.
    """
    original_shape = arr.shape
    if len(original_shape) == 2:
        arr = arr.reshape(1, *original_shape)
    elif len(original_shape) == 1:
        arr = arr.reshape(1, original_shape[0], 1)
    if shift > 0:
        shifted = np.pad(arr, ((0, 0), (shift, 0), (0, 0)), mode='constant')[:, :-shift, :]
    else:
        shifted = np.pad(arr, ((0, 0), (0, -shift), (0, 0)), mode='constant')[:, -shift:, :]
    if len(original_shape) != 3:
        shifted = shifted.reshape(original_shape)
    return shifted
