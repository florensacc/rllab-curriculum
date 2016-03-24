from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.env_spec import EnvSpec
from rllab.core.serializable import Serializable
from rllab.spaces import Product


class SubgoalEnv(ProxyEnv, Serializable):

    def __init__(
            self,
            wrapped_env,
            subgoal_space,
            low_obs_action_history=False,
            low_action_history_length=0):
        super(SubgoalEnv, self).__init__(wrapped_env=wrapped_env)
        Serializable.quick_init(self, locals())
        self._subgoal_space = subgoal_space
        self._low_obs_action_history = low_obs_action_history
        self._low_action_history_length = low_action_history_length

    @property
    def subgoal_space(self):
        return self._subgoal_space

    @property
    def spec(self):
        return SubgoalEnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
            subgoal_space=self.subgoal_space,
            low_obs_action_history=self._low_obs_action_history,
            low_action_history_length=self._low_action_history_length
        )


class SubgoalEnvSpec(EnvSpec):

    def __init__(
            self,
            observation_space,
            action_space,
            subgoal_space,
            low_obs_action_history=False,
            low_action_history_length=0):
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
