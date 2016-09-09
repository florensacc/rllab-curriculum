


import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np

from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.spaces import Box

from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import ext
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
import theano.tensor as TT


class DuelTwoPartGaussianMLPPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(self, env_spec, master_policy, share_std_layers=False):
        """
        :type master_policy: TwoPartGaussianMLPPolicy
        :param share_std_layers: whether to share the std layers with other policies
        """
        Serializable.quick_init(self, locals())
        self.master_policy = master_policy
        self.share_std_layers = share_std_layers

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim
        subgoal_dim = self.master_policy.subgoal_dim
        subgoal_hidden_sizes = self.master_policy.subgoal_hidden_sizes
        std_hidden_sizes = self.master_policy.std_hidden_sizes
        hidden_nonlinearity = self.master_policy.hidden_nonlinearity
        subgoal_network = MLP(
            input_shape=(obs_dim,),
            output_dim=subgoal_dim,
            hidden_sizes=subgoal_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=hidden_nonlinearity,
        )
        l_obs = subgoal_network.input_layer
        l_subgoal = subgoal_network.output_layer
        if share_std_layers:
            l_action_log_std = master_policy.l_action_log_std  # l_log_std
        else:
            if master_policy.adaptive_std:
                if master_policy.std_share_network:
                    l_action_log_std = L.DenseLayer(
                        master_policy.action_mean_network.layers[-2],
                        num_units=action_dim,
                        nonlinearity=None,
                    )
                else:
                    action_log_std_network = MLP(
                        input_layer=L.concat([master_policy.l_obs, master_policy.l_subgoal]),
                        output_dim=action_dim,
                        hidden_sizes=std_hidden_sizes,
                        hidden_nonlinearity=hidden_nonlinearity,
                        output_nonlinearity=None,
                        name="action_log_std_network",
                    )
                    l_action_log_std = action_log_std_network.output_layer
            else:
                l_action_log_std = ParamLayer(
                    master_policy.l_obs,
                    num_units=action_dim,
                    param=lasagne.init.Constant(np.log(master_policy.init_std)),
                    name="action_log_std",
                    trainable=master_policy.learn_std,
                )
        self.l_subgoal = l_subgoal
        self.l_obs = l_obs
        self.l_action_mean = master_policy.l_action_mean
        self.l_action_log_std = l_action_log_std

        dist_info_sym = self.dist_info_sym(obs_var=l_obs.input_var)

        StochasticPolicy.__init__(self, env_spec)
        LasagnePowered.__init__(self, [self.l_subgoal, self.l_action_mean, self.l_action_log_std])

        self.f_action_dist = ext.compile_function(
            [l_obs.input_var],
            [dist_info_sym["mean"], dist_info_sym["log_std"]],
        )

    def dist_info_sym(self, obs_var, state_info_vars=None):
        subgoal_var = L.get_output(self.l_subgoal, {self.l_obs: obs_var})
        action_mean_var, action_log_std_var = L.get_output(
            [self.l_action_mean, self.l_action_log_std],
            {self.master_policy.l_obs: obs_var, self.master_policy.l_subgoal: subgoal_var},
        )
        if self.master_policy.min_std is not None:
            action_log_std_var = TT.maximum(action_log_std_var, np.log(self.master_policy.min_std))
        return dict(mean=action_mean_var, log_std=action_log_std_var)

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        mean, log_std = [x[0] for x in self.f_action_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self.master_policy.action_dist


class TwoPartGaussianMLPPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            subgoal_dim,
            subgoal_hidden_sizes=(32, 32),
            action_hidden_sizes=(32, 32),
            learn_std=True,
            init_std=1.0,
            min_std=1e-6,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=None,
    ):
        """
        :param env_spec:
        :param subgoal_dim: dimension of subgoal
        :param subgoal_hidden_sizes: hidden sizes of subgoal network
        :param action_hidden_sizes: hidden sizes of action network
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :return:
        """
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        subgoal_network = MLP(
            input_shape=(obs_dim,),
            output_dim=subgoal_dim,
            hidden_sizes=subgoal_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=hidden_nonlinearity,
            name="subgoal_network"
        )
        l_obs = subgoal_network.input_layer
        l_subgoal = subgoal_network.output_layer
        # create network
        action_mean_network = MLP(
            input_layer=L.concat([l_obs, l_subgoal]),
            output_dim=action_dim,
            hidden_sizes=action_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
            name="action_mean_network"
        )

        l_action_mean = action_mean_network.output_layer
        obs_var = l_obs.input_var

        if adaptive_std:
            if std_share_network:
                # if share network, only add a linear layer parallel to the last layer of action mean network
                l_action_log_std = L.DenseLayer(
                    action_mean_network.layers[-2],
                    num_units=action_dim,
                    nonlinearity=None,
                )
            else:
                action_log_std_network = MLP(
                    input_layer=L.concat([l_obs, l_subgoal]),
                    output_dim=action_dim,
                    hidden_sizes=std_hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=None,
                    name="action_log_std_network",
                )
                l_action_log_std = action_log_std_network.output_layer
        else:
            l_action_log_std = ParamLayer(
                l_obs,
                num_units=action_dim,
                param=lasagne.init.Constant(np.log(init_std)),
                name="action_log_std",
                trainable=learn_std,
            )

        action_mean_var, action_log_std_var = L.get_output([l_action_mean, l_action_log_std])

        if min_std is not None:
            action_log_std_var = TT.maximum(action_log_std_var, np.log(min_std))

        self.action_mean_var, self.action_log_std_var = action_mean_var, action_log_std_var

        self.l_subgoal = l_subgoal
        self.l_action_mean = l_action_mean
        self.l_action_log_std = l_action_log_std
        self.l_obs = l_obs
        self.adaptive_std = adaptive_std
        self.subgoal_dim = subgoal_dim
        self.subgoal_hidden_sizes = subgoal_hidden_sizes
        self.std_share_network = std_share_network
        self.std_hidden_sizes = std_hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.action_mean_network = action_mean_network
        self.min_std = min_std
        self.init_std = init_std
        self.learn_std = learn_std

        self.action_dist = DiagonalGaussian(action_dim)

        LasagnePowered.__init__(self, [l_subgoal, l_action_mean, l_action_log_std])
        super(TwoPartGaussianMLPPolicy, self).__init__(env_spec)

        self.f_action_dist = ext.compile_function(
            inputs=[obs_var],
            outputs=[action_mean_var, action_log_std_var],
        )

    def dist_info_sym(self, obs_var, state_info_vars=None):
        action_mean_var, action_log_std_var = L.get_output([self.l_action_mean, self.l_action_log_std], obs_var)
        if self.min_std is not None:
            action_log_std_var = TT.maximum(action_log_std_var, np.log(self.min_std))
        return dict(mean=action_mean_var, log_std=action_log_std_var)

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        mean, log_std = [x[0] for x in self.f_action_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self.action_dist
