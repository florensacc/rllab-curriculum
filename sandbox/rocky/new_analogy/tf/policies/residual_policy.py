from rllab.core.serializable import Serializable
from sandbox.rocky.tf.policies.base import Policy


class ResidualPolicy(Policy, Serializable):
    def __init__(self, env_spec, wrapped_policy):
        Serializable.quick_init(self, locals())
        self.wrapped_policy = wrapped_policy
        Policy.__init__(self, env_spec)

    def dist_info_sym(self, obs_var, state_info_vars=None):
        dist_info = self.wrapped_policy.dist_info_sym(obs_var)
        return dict(
            dist_info,
            mean=dist_info["mean"] + obs_var[:, :self.action_space.flat_dim]
        )

    def get_params_internal(self, **tags):
        return self.wrapped_policy.get_params(**tags)

    def get_actions(self, observations):
        actions, agent_infos = self.wrapped_policy.get_actions(observations)
        flat_obs = self.observation_space.flatten_n(observations)
        actions += flat_obs[:, :self.action_space.flat_dim]
        agent_infos["mean"] += flat_obs[:, :self.action_space.flat_dim]
        return actions, agent_infos

    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    @property
    def distribution(self):
        return self.wrapped_policy.distribution

    def log_diagnostics(self, paths):
        self.wrapped_policy.log_diagnostics(paths)

    @property
    def vectorized(self):
        return self.wrapped_policy.vectorized

    def reset(self, dones=None):
        self.wrapped_policy.reset(dones=dones)
