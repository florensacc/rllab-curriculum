import numpy as np
from cached_property import cached_property

from sandbox.rocky.chainer.core.link_powered import LinkPowered
from sandbox.rocky.chainer.core.network import MLP
from sandbox.rocky.chainer.distributions.diagonal_gaussian import DiagonalGaussian
from sandbox.rocky.chainer.misc import tensor_utils
# from sandbox.rocky.chainer.spaces import Box
from sandbox.rocky.chainer.policies.base import StochasticPolicy
import chainer
import chainer.functions as F
import chainer.links as L
from rllab.misc import logger

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides


class GaussianMLPActorCritic(StochasticPolicy, LinkPowered, chainer.Chain, Serializable):
    def __init__(
            self,
            env_spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=F.tanh,
    ):
        Serializable.quick_init(self, locals())
        StochasticPolicy.__init__(self, env_spec)
        chainer.Chain.__init__(self)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        head_network = MLP(
            input_shape=(obs_dim,),
            output_dim=action_dim + 1,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=None,
        )

        l_log_std = L.Parameter(array=np.zeros(action_dim, dtype=np.float32))

        self.add_link("head_network", head_network)
        self.add_link("l_log_std", l_log_std)

        self.action_dim = action_dim

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars=None):
        obs_var = F.cast(obs_var, np.float32)
        head_out = self.head_network(obs_var)
        N = obs_var.shape[0]
        means = head_out[:, :self.action_dim]
        vf = head_out[:, self.action_dim:]
        log_stds = F.tile(F.expand_dims(self.l_log_std(), 0), (N, 1))
        return dict(
            mean=means, log_std=log_stds, vf=vf
        )

    @property
    def vectorized(self):
        return True

    @overrides
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        dist_info = {k: v.data for k, v in self.dist_info_sym(flat_obs).items()}
        mean = dist_info["mean"]
        log_std = dist_info["log_std"]
        actions = np.random.normal(size=mean.shape) * np.exp(log_std) + mean
        return actions, dist_info

    def get_actions_sym(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        dist_info_vars = self.dist_info_sym(flat_obs)
        actions = self.distribution.sample_sym(dist_info_vars)
        return actions, dist_info_vars

    @cached_property
    def distribution(self):
        return DiagonalGaussian(self.action_dim)

    def log_diagnostics(self, paths):
        if len(paths) > 0:
            log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
            logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))
        else:
            logger.record_tabular('AveragePolicyStd', np.nan)
