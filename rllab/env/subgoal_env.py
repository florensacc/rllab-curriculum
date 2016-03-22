from rllab.env.proxy_env import ProxyEnv
from rllab.env.env_spec import EnvSpec
from rllab.core.serializable import Serializable
from rllab.spaces import Product


class SubgoalEnv(ProxyEnv, Serializable):

    def __init__(
            self,
            wrapped_env,
            subgoal_space):
        super(SubgoalEnv, self).__init__(wrapped_env=wrapped_env)
        Serializable.quick_init(self, locals())
        self._subgoal_space = subgoal_space

    @property
    def subgoal_space(self):
        return self.subgoal_space

    @property
    def spec(self):
        return SubgoalEnvSpec(
            observation_space=self.wrapped_env.observation_space,
            action_space=self.wrapped_env.action_space,
            subgoal_space=self.subgoal_space,
        )


class SubgoalEnvSpec(EnvSpec):

    def __init__(
            self,
            observation_space,
            action_space,
            subgoal_space):
        Serializable.quick_init(self, locals())
        super(SubgoalEnvSpec, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
        )
        self._subgoal_space = subgoal_space
        self._high_env_spec = EnvSpec(
            observation_space=observation_space,
            action_space=subgoal_space,
        )
        self._low_env_spec = EnvSpec(
            observation_space=Product(observation_space, subgoal_space),
            action_space=action_space,
        )

    @property
    def subgoal_space(self):
        return self._subgoal_space

    @property
    def high_env_spec(self):
        return self._high_env_spec

    @property
    def low_env_spec(self):
        return self._low_env_spec
