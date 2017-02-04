from rllab.core.serializable import Serializable
from sandbox.rocky.tf.policies.base import Policy


class DeterministicPolicy(Policy, Serializable):
    def __init__(self, env_spec, wrapped_policy):
        Serializable.quick_init(self, locals())
        self.wrapped_policy = wrapped_policy
        Policy.__init__(self, env_spec)

    def get_params_internal(self, **tags):
        return self.wrapped_policy.get_params(**tags)

    def get_actions(self, observations):
        if hasattr(self.wrapped_policy, "get_greedy_actions"):
            return self.wrapped_policy.get_greedy_actions(observations)
        _, agent_infos = self.wrapped_policy.get_actions(observations)
        return self.wrapped_policy.distribution.maximum_a_posteriori(agent_infos), agent_infos

    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def reset(self, dones=None):
        self.wrapped_policy.reset(dones=dones)

    @property
    def vectorized(self):
        return self.wrapped_policy.vectorized
