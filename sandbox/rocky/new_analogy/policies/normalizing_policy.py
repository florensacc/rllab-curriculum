from rllab.core.serializable import Serializable
from sandbox.rocky.tf.policies.base import Policy
import numpy as np
import tensorflow as tf

from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian


class NormalizingPolicy(Policy, Serializable):
    def __init__(self, wrapped_policy, paths=None, obs_mean=None, obs_std=None,
                 action_mean=None, action_std=None, normalize_obs=False, normalize_actions=True):
        self.obs_dim = wrapped_policy.observation_space.flat_dim
        self.action_dim = wrapped_policy.action_space.flat_dim

        obs = np.concatenate([p["observations"] for p in paths], axis=0)
        actions = np.concatenate([p["actions"] for p in paths], axis=0)

        if normalize_obs:
            obs_mean = np.mean(obs, axis=0, keepdims=True)
            obs_std = np.std(obs, axis=0, keepdims=True) + 1e-5
        else:
            obs_mean = np.zeros_like(obs[0])
            obs_std = np.ones_like(obs[0])

        if normalize_actions:
            action_mean = np.mean(actions, axis=0, keepdims=True)
            action_std = np.std(actions, axis=0, keepdims=True) + 1e-5
        else:
            action_mean = np.zeros_like(actions[0])
            action_std = np.ones_like(actions[0])

        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.action_mean = action_mean
        self.action_std = action_std

        paths = None
        Serializable.quick_init(self, locals())

        self.obs_mean_var = tf.Variable(self.obs_mean, dtype=tf.float32, name="obs_mean")
        self.obs_std_var = tf.Variable(self.obs_std, dtype=tf.float32, name="obs_std")
        self.action_mean_var = tf.Variable(self.action_mean, dtype=tf.float32, name="action_mean")
        self.action_std_var = tf.Variable(self.action_std, dtype=tf.float32, name="action_std")
        self.wrapped_policy = wrapped_policy

        Policy.__init__(self, self.wrapped_policy.env_spec)

    def dist_info_sym(self, obs_var, state_info_vars, **kwargs):
        norm_obs_var = (obs_var - self.obs_mean_var) / self.obs_std_var
        dist_info = self.wrapped_policy.dist_info_sym(
            norm_obs_var, state_info_vars=state_info_vars,
            **kwargs
        )
        if isinstance(self.distribution, DiagonalGaussian):
            dist_info["mean"] = dist_info["mean"] * self.action_std_var + self.action_mean_var
            dist_info["log_std"] = dist_info["log_std"] + tf.log(self.action_std_var + 1e-8)
        else:
            raise NotImplementedError
        return dist_info

    @property
    def vectorized(self):
        return self.wrapped_policy.vectorized

    @property
    def distribution(self):
        return self.wrapped_policy.distribution

    def log_diagnostics(self, paths):
        self.wrapped_policy.log_diagnostics(paths)

    @property
    def state_info_specs(self):
        return self.wrapped_policy.state_info_specs

    def get_action(self, obs):
        actions, agent_infos = self.get_actions([obs])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        obs_space = self.observation_space
        action_space = self.action_space
        flat_obs = obs_space.flatten_n(observations)
        flat_normalized_obs = (flat_obs - self.obs_mean) / self.obs_std
        normalized_obs = obs_space.unflatten_n(flat_normalized_obs)
        actions, agent_infos = self.wrapped_policy.get_actions(normalized_obs)
        normalized_action = action_space.flatten_n(actions)
        actions = action_space.unflatten_n(normalized_action * self.action_std + self.action_mean)
        return actions, agent_infos

    def get_params_internal(self, **tags):
        params = list(self.wrapped_policy.get_params_internal(**tags))
        if not tags.get('trainable', False) and not tags.get('regularizable', False):
            params.extend([self.obs_mean_var, self.obs_std_var, self.action_mean_var, self.action_std_var])
        return params

    def set_param_values(self, flattened_params, **tags):
        Policy.set_param_values(self, flattened_params, **tags)
        self.obs_mean, self.obs_std, self.action_mean, self.action_std = tf.get_default_session().run([
            self.obs_mean_var, self.obs_std_var, self.action_mean_var, self.action_std_var
        ])

    def reset(self, dones=None):
        self.wrapped_policy.reset(dones=dones)
