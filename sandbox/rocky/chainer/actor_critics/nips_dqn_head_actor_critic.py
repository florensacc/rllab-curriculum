import chainer
from cached_property import cached_property

from rllab.core.serializable import Serializable
from sandbox.rocky.chainer.core.link_powered import LinkPowered
from sandbox.rocky.chainer.distributions.categorical import Categorical
from sandbox.rocky.chainer.policies.base import StochasticPolicy
import chainer.links as L
import chainer.functions as F
import numpy as np
from rllab.misc import logger


def init_like_torch(link):
    # Mimic torch's default parameter initialization
    # Each pre-activation unit has variance 1
    # TODO(muupan): Use chainer's initializers when it is merged
    for l in link.links():
        if isinstance(l, L.Linear):
            out_channels, in_channels = l.W.data.shape
            stdv = 1 / np.sqrt(in_channels)
            l.W.data[:] = np.random.uniform(-stdv, stdv, size=l.W.data.shape).astype(np.float32)
            if l.b is not None:
                l.b.data[:] = np.random.uniform(-stdv, stdv,
                                                size=l.b.data.shape).astype(np.float32)
        elif isinstance(l, L.Convolution2D):
            out_channels, in_channels, kh, kw = l.W.data.shape
            stdv = 1 / np.sqrt(in_channels * kh * kw)
            l.W.data[:] = np.random.uniform(-stdv, stdv, size=l.W.data.shape).astype(np.float32)
            if l.b is not None:
                l.b.data[:] = np.random.uniform(-stdv, stdv,
                                                size=l.b.data.shape).astype(np.float32)


class NIPSDQNHeadActorCritic(StochasticPolicy, LinkPowered, chainer.Chain, Serializable):
    def __init__(
            self,
            env_spec,
    ):
        Serializable.quick_init(self, locals())
        StochasticPolicy.__init__(self, env_spec)
        chainer.Chain.__init__(self)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        l_conv1 = L.Convolution2D(4, 16, 8, stride=4, bias=0.1)
        l_conv2 = L.Convolution2D(16, 32, 4, stride=2, bias=0.1)
        l_fc1 = L.Linear(2592, 256, bias=0.1)

        l_logits = L.Linear(256, action_dim)
        l_vf = L.Linear(256, 1)

        self.add_link("l_conv1", l_conv1)
        self.add_link("l_conv2", l_conv2)
        self.add_link("l_fc1", l_fc1)
        self.add_link("l_logits", l_logits)
        self.add_link("l_vf", l_vf)

        init_like_torch(self)

        self.action_dim = action_dim

    def dist_info_sym(self, obs_var, state_info_vars=None):
        obs_var = F.cast(obs_var, np.float32)
        h = obs_var
        h = F.relu(self.l_conv1(h))
        h = F.relu(self.l_conv2(h))
        h = F.relu(self.l_fc1(h))
        prob = F.softmax(self.l_logits(h))
        vf = self.l_vf(h)
        return dict(
            prob=prob, vf=vf,
        )

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        actions_var, dist_info_vars = self.get_actions_sym(observations)
        return actions_var.data, {k: v.data for k, v in dist_info_vars.items()}

    def get_action_sym(self, observation):
        observations = (np.asarray([observation]) + 1.) * 0.5
        dist_info_vars = self.dist_info_sym(observations)
        actions = self.distribution.sample_sym(dist_info_vars)
        return actions[0], {k: v[0] for k, v in dist_info_vars.items()}

    def get_actions_sym(self, observations):
        observations = (np.asarray(observations) + 1.) * 0.5
        dist_info_vars = self.dist_info_sym(observations)
        actions = self.distribution.sample_sym(dist_info_vars)
        return actions, dist_info_vars

    @cached_property
    def distribution(self):
        return Categorical(self.action_dim)


    # def apply_normalization(self):
