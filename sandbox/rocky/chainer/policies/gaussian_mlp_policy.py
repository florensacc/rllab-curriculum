from sandbox.rocky.chainer.core.link_powered import LinkPowered
from sandbox.rocky.chainer.distributions.diagonal_gaussian import DiagonalGaussian
from sandbox.rocky.chainer.policies.base import StochasticPolicy
from sandbox.rocky.chainer.core.network import MLP
import chainer
import chainer.links as L
import numpy as np
import chainer.functions as F
from rllab.misc import logger


class GaussianMLPPolicy(StochasticPolicy, LinkPowered, chainer.Chain):
    def __init__(
            self,
            env_spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=F.Tanh(),
            output_nonlinearity=None,
    ):
        StochasticPolicy.__init__(self, env_spec=env_spec)
        chainer.Chain.__init__(self)
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        # create network
        mean_network = MLP(
            input_shape=(obs_dim,),
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
        )
        l_log_std = L.Parameter(array=np.zeros((action_dim,), dtype=np.float32))
        self.add_link("mean_network", mean_network)
        self.add_link("l_log_std", l_log_std)

    def reset(self, dones=None):
        pass

    def dist_info_sym(self, obs_var, state_info_vars):
        mean = self.mean_network(F.cast(obs_var, np.float32))
        log_std = F.tile(F.expand_dims(self.l_log_std(), 0), (len(obs_var.data), 1))
        return dict(mean=mean, log_std=log_std)

    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        mean = self.mean_network(np.asarray([flat_obs], dtype=np.float32)).data[0]
        log_std = self.l_log_std().data
        action = np.random.normal(size=mean.shape) * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        mean = self.mean_network(np.asarray(flat_obs, dtype=np.float32)).data
        log_std = np.tile(np.expand_dims(self.l_log_std().data, 0), (len(observations), 1))
        actions = np.random.normal(size=mean.shape) * np.exp(log_std) + mean
        return actions, dict(mean=mean, log_std=log_std)

    @property
    def vectorized(self):
        return True

    @property
    def recurrent(self):
        return False

    @property
    def distribution(self):
        return DiagonalGaussian(self.action_space.flat_dim)

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))
