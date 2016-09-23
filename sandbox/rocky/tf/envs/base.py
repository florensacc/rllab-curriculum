from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.base import EnvSpec
from rllab.spaces.box import Box as TheanoBox
from rllab.spaces.discrete import Discrete as TheanoDiscrete
from rllab.spaces.product import Product as TheanoProduct
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
    @property
    def observation_space(self):
        return to_tf_space(self.wrapped_env.observation_space)

    @property
    def action_space(self):
        return to_tf_space(self.wrapped_env.action_space)

    @property
    def spec(self):
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

    @classmethod
    def wrap(cls, env_cls, **extra_kwargs):
        # Use a class wrapper rather than a lambda method for smoother serialization
        return WrappedCls(cls, env_cls, extra_kwargs)
