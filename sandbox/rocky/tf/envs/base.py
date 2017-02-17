from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.base import EnvSpec
from rllab.spaces.box import Box as TheanoBox
from rllab.spaces.discrete import Discrete as TheanoDiscrete
from rllab.spaces.product import Product as TheanoProduct
from sandbox.rocky.tf.envs.vec_env import VecEnv
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.spaces.product import Product
from cached_property import cached_property


def to_tf_space(space):
    if isinstance(space, TheanoBox):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, TheanoDiscrete):
        return Discrete(space.n)
    elif isinstance(space, TheanoProduct):
        return Product(list(map(to_tf_space, space.components)))
    else:
        raise NotImplementedError


class WrappedCls(object):
    def __init__(self, cls, env_cls, extra_kwargs):
        self.cls = cls
        self.env_cls = env_cls
        self.extra_kwargs = extra_kwargs

    def __call__(self, *args, **kwargs):
        return self.cls(self.env_cls(*args, **dict(self.extra_kwargs, **kwargs)))


class TfEnv(ProxyEnv):
    @cached_property
    def observation_space(self):
        return to_tf_space(self.wrapped_env.observation_space)

    @cached_property
    def action_space(self):
        return to_tf_space(self.wrapped_env.action_space)

    @cached_property
    def spec(self):
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

    @property
    def vectorized(self):
        return getattr(self.wrapped_env, "vectorized", False)

    def vec_env_executor(self, n_envs):
        return VecTfEnv(self.wrapped_env.vec_env_executor(n_envs=n_envs))

    @classmethod
    def wrap(cls, env_cls, **extra_kwargs):
        # Use a class wrapper rather than a lambda method for smoother serialization
        return WrappedCls(cls, env_cls, extra_kwargs)

    def get_current_obs(self):
        return self.wrapped_env.get_current_obs()

class VecTfEnv(VecEnv):

    def __init__(self, vec_env):
        self.vec_env = vec_env

    def reset_trial(self, dones, seeds=None, *args, **kwargs):
        return self.vec_env.reset_trial(dones, seeds=seeds, *args, **kwargs)

    def reset(self, dones, seeds=None, *args, **kwargs):
        return self.vec_env.reset(dones, seeds=seeds, *args, **kwargs)

    @property
    def n_envs(self):
        return self.vec_env.n_envs

    def step(self, action_n, max_path_length=None):
        return self.vec_env.step(action_n, max_path_length)

    def terminate(self):
        self.vec_env.terminate()

    def handle_policy_reset(self, policy, dones):
        self.vec_env.handle_policy_reset(policy, dones)
