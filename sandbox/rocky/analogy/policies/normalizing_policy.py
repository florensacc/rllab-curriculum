from sandbox.rocky.tf.core.parameterized import Parameterized
from sandbox.rocky.tf.policies.base import Policy
from rllab.core.serializable import Serializable
import tensorflow as tf
import numpy as np


class NormalizingPolicy(Policy, Serializable):
    def __init__(self, wrapped_policy, demo_paths=None, analogy_paths=None, obs_mean=None, obs_std=None,
                 action_mean=None, action_std=None, normalize_obs=False, normalize_actions=True):
        self.obs_dim = wrapped_policy.observation_space.flat_dim
        self.action_dim = wrapped_policy.action_space.flat_dim

        if demo_paths is not None:
            all_paths = list(demo_paths) + list(analogy_paths)

            obs = np.concatenate([p["observations"] for p in all_paths], axis=0)
            actions = np.concatenate([p["actions"] for p in all_paths], axis=0)

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

        demo_paths = None
        analogy_paths = None
        Serializable.quick_init(self, locals())

        self.obs_mean_var = tf.Variable(self.obs_mean, dtype=tf.float32, name="obs_mean")
        self.obs_std_var = tf.Variable(self.obs_std, dtype=tf.float32, name="obs_std")
        self.action_mean_var = tf.Variable(self.action_mean, dtype=tf.float32, name="action_mean")
        self.action_std_var = tf.Variable(self.action_std, dtype=tf.float32, name="action_std")
        self.wrapped_policy = wrapped_policy

        Policy.__init__(self, self.wrapped_policy.env_spec)

    def action_sym(self, analogy_obs_var, state_info_vars, **kwargs):
        demo_obs_var = state_info_vars["demo_obs"]
        demo_actions_var = state_info_vars["demo_actions"]
        demo_valids_var = state_info_vars["demo_valids"]
        norm_obs_var = (analogy_obs_var - self.obs_mean_var) / self.obs_std_var
        norm_demo_obs_var = (demo_obs_var - self.obs_mean_var) / self.obs_std_var
        norm_demo_action_var = (demo_actions_var - self.action_mean_var) / self.action_std_var
        norm_action_var = self.wrapped_policy.action_sym(
            norm_obs_var, state_info_vars=dict(
                state_info_vars,
                demo_obs=norm_demo_obs_var,
                demo_actions=norm_demo_action_var,
                demo_valids_var=demo_valids_var,
            ), **kwargs
        )
        return norm_action_var * self.action_std_var + self.action_mean_var

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

    def apply_demo(self, path):
        self.apply_demos([path])

    def apply_demos(self, paths):
        morphed_paths = []
        for path in paths:
            # only process it if necessary
            demo_obs = path["observations"]
            demo_actions = path["actions"]
            norm_demo_obs = (demo_obs - self.obs_mean) / self.obs_std
            norm_demo_actions = (demo_actions - self.action_mean) / self.action_std
            morphed_paths.append(dict(path, observations=norm_demo_obs, actions=norm_demo_actions))
        self.wrapped_policy.apply_demos(morphed_paths)
