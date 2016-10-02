from rllab.core.serializable import Serializable
from rllab.sampler.utils import rollout


class PolicyDemoCollector(Serializable):
    def __init__(self, policy_cls):
        Serializable.quick_init(self, locals())
        self.policy_cls = policy_cls

    def collect_demo(self, env, horizon):
        policy = self.policy_cls(env)
        return rollout(env, policy, max_path_length=horizon)
