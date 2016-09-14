from sandbox.rocky.tf.policies.base import Policy
from rllab.core.serializable import Serializable


class ApplyDemoPolicy(Policy, Serializable):

    def __init__(self, wrapped_policy, demo_path):
        Serializable.quick_init(self, locals())
        Policy.__init__(self, wrapped_policy.env_spec)
        self.wrapped_policy = wrapped_policy
        self.demo_path = demo_path

    def get_action(self, obs):
        return self.wrapped_policy.get_action(obs)

    def get_params_internal(self, **tags):
        return self.wrapped_policy.get_params_internal(**tags)

    def set_param_values(self, flattened_params, **tags):
        self.wrapped_policy.set_param_values(flattened_params, **tags)

    def reset(self, dones=None):
        self.wrapped_policy.reset(dones=dones)
        self.wrapped_policy.apply_demo(self.demo_path)

