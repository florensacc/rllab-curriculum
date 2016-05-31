from __future__ import print_function
from __future__ import absolute_import

from rllab.policies.base import StochasticPolicy
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.spaces.discrete import Discrete
from rllab.distributions.recurrent_categorical import RecurrentCategorical
from rllab.distributions.recurrent_diagonal_gaussian import RecurrentDiagonalGaussian
from rllab.misc import tensor_utils
from rllab.envs.base import EnvSpec
from rllab.misc import ext
from rllab.misc import special
from rllab.core.network import MLP
from rllab.core.lasagne_layers import OpLayer
import lasagne.layers as L
import lasagne.nonlinearities as NL
import theano
import theano.tensor as TT
import numpy as np


class ContinuousRNNPolicy(StochasticPolicy, LasagnePowered, Serializable):
    """
    Structure the hierarchical policy as a recurrent network with continuous stochastic hidden units, which will play
    the role of internal goals.
    """

    def __init__(self,
                 env_spec,
                 hidden_state_dim,
                 bottleneck_dim,
                 hidden_sizes=(32, 32),
                 use_decision_nodes=True,
                 hid_hidden_sizes=None,
                 decision_hidden_sizes=None,
                 action_hidden_sizes=None,
                 bottleneck_hidden_sizes=None,
                 hidden_nonlinearity=TT.tanh):
        """
        :type env_spec: EnvSpec
        :param use_decision_nodes: whether to have decision units, which governs whether the subgoals should be
        resampled
        :param random_reset: whether to randomly set the first subgoal
        """
        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Discrete)

        if hid_hidden_sizes is None:
            hid_hidden_sizes = hidden_sizes
        if decision_hidden_sizes is None:
            decision_hidden_sizes = hidden_sizes
        if action_hidden_sizes is None:
            action_hidden_sizes = hidden_sizes
        if bottleneck_hidden_sizes is None:
            bottleneck_hidden_sizes = hidden_sizes

        self.hidden_state = None
        self.hidden_state_dim = hidden_state_dim
        self.use_decision_nodes = use_decision_nodes
        self.bottleneck_dim = bottleneck_dim

        l_prev_hidden = L.InputLayer(
            shape=(None, hidden_state_dim),
            name="prev_hidden",
        )
        l_hidden = L.InputLayer(
            shape=(None, hidden_state_dim),
            name="hidden",
        )
        l_raw_obs = L.InputLayer(
            shape=(None, env_spec.observation_space.flat_dim),
            name="obs",
        )

        bottleneck_mean_network = MLP(
            input_layer=L.concat([l_raw_obs, l_prev_hidden], name="bottleneck_mean_network_input"),
            hidden_sizes=bottleneck_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=None,
            output_dim=bottleneck_dim,
            name="bottleneck_mean_network"
        )
        bottleneck_log_std_network = MLP(
            input_layer=L.concat([l_raw_obs, l_prev_hidden], name="bottleneck_log_std_network_input"),
            hidden_sizes=bottleneck_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=None,
            output_dim=bottleneck_dim,
            name="bottleneck_log_std_network"
        )

        l_bottleneck_mean = bottleneck_mean_network.output_layer
        l_bottleneck_log_std = bottleneck_log_std_network.output_layer

        l_bottleneck = L.InputLayer(
            shape=(None, bottleneck_dim),
            name="bottleneck"
        )

        l_obs = l_bottleneck

        hidden_mean_network = MLP(
            input_layer=L.concat([l_obs, l_prev_hidden], name="hidden_mean_network_input"),
            hidden_sizes=hid_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=None,
            output_dim=hidden_state_dim,
            name="hidden_mean_network"
        )
        hidden_log_std_network = MLP(
            input_layer=L.concat([l_obs, l_prev_hidden], name="hidden_log_std_network_input"),
            hidden_sizes=hid_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=None,
            output_dim=hidden_state_dim,
            name="hidden_log_std_network"
        )
        action_network = MLP(
            input_layer=L.concat([l_obs, l_hidden], name="action_network_input"),
            hidden_sizes=action_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=TT.nnet.softmax,  # tf.nn.softmax,
            output_dim=env_spec.action_space.n,
            name="action_network"
        )

        l_hidden_mean = hidden_mean_network.output_layer
        l_hidden_log_std = hidden_log_std_network.output_layer

        l_action_prob = action_network.output_layer

        self.f_hidden_dist = ext.compile_function(
            [l_obs.input_var, l_prev_hidden.input_var],
            L.get_output([l_hidden_mean, l_hidden_log_std]),
        )
        self.f_action_prob = ext.compile_function(
            [l_obs.input_var, l_hidden.input_var],
            L.get_output(l_action_prob),
        )
        self.f_bottleneck_dist = ext.compile_function(
            [l_raw_obs.input_var, l_prev_hidden.input_var],
            L.get_output([l_bottleneck_mean, l_bottleneck_log_std]),
        )

        StochasticPolicy.__init__(self, env_spec=env_spec)

        used_layers = [l_hidden_mean, l_hidden_log_std, l_action_prob, l_bottleneck_mean, l_bottleneck_log_std]
        LasagnePowered.__init__(self, used_layers)

        self.l_hidden_mean = l_hidden_mean
        self.l_hidden_log_std = l_hidden_log_std
        self.l_action_prob = l_action_prob
        self.l_raw_obs = l_raw_obs
        self.l_obs = l_obs
        self.l_prev_hidden = l_prev_hidden
        self.l_bottleneck = l_bottleneck
        self.l_bottleneck_mean = l_bottleneck_mean
        self.l_bottleneck_log_std = l_bottleneck_log_std
        self.l_hidden = l_hidden

        self.hidden_dist = RecurrentDiagonalGaussian(self.hidden_state_dim)
        self.action_dist = RecurrentCategorical(self.action_space.n)
        self.bottleneck_dist = RecurrentDiagonalGaussian(self.bottleneck_dim)

    @property
    def distribution(self):
        return self

    @property
    def recurrent(self):
        return True

    @property
    def dist_info_keys(self):
        return [k for k, _ in self.dist_info_specs]

    @property
    def state_info_keys(self):
        return [k for k, _ in self.state_info_specs]

    @property
    def state_info_specs(self):
        specs = [
            # ("hidden_mean", (self.hidden_state_dim,)),
            # ("hidden_log_std", (self.hidden_state_dim,)),
            ("hidden_epsilon", (self.hidden_state_dim,)),
            # ("prev_hidden", (self.hidden_state_dim,)),
            ("bottleneck_epsilon", (self.bottleneck_dim,))
        ]
        return specs

    @property
    def dist_info_specs(self):
        specs = [
            ("action_prob", (self.action_space.n,)),
            # ("hidden_mean", (self.hidden_state_dim,)),
            # ("hidden_log_std", (self.hidden_state_dim,)),
            # ("bottleneck_mean", (self.bottleneck_dim,)),
            # ("bottleneck_log_std", (self.bottleneck_dim,)),
        ]
        return specs

    def dist_info_sym(self, obs_var, state_info_vars):
        # obs: N * T * S
        # prev_hidden_var = state_info_vars["prev_hidden"]
        bottleneck_epsilon = state_info_vars["bottleneck_epsilon"]
        hidden_epsilon = state_info_vars["hidden_epsilon"]

        N = obs_var.shape[0]

        def rnn_step(cur_obs, cur_bottleneck_epsilon, cur_hidden_epsilon, prev_hidden, prev_action_prob):
            bottleneck_mean_var, bottleneck_log_std_var = L.get_output(
                [self.l_bottleneck_mean, self.l_bottleneck_log_std],
                inputs={self.l_raw_obs: cur_obs, self.l_prev_hidden: prev_hidden}
            )
            bottleneck = cur_bottleneck_epsilon * TT.exp(bottleneck_log_std_var) + bottleneck_mean_var
            hidden_mean_var, hidden_log_std_var = L.get_output(
                [self.l_hidden_mean, self.l_hidden_log_std],
                inputs={self.l_obs: bottleneck, self.l_prev_hidden: prev_hidden}
            )
            hidden = cur_hidden_epsilon * TT.exp(hidden_log_std_var) + hidden_mean_var
            action_prob_var = L.get_output(self.l_action_prob, inputs={self.l_obs: bottleneck, self.l_hidden: hidden})
            return hidden, action_prob_var

        all_hidden, all_action_prob = theano.scan(
            rnn_step,
            sequences=[
                obs_var.dimshuffle(1, 0, 2),
                bottleneck_epsilon.dimshuffle(1, 0, 2),
                hidden_epsilon.dimshuffle(1, 0, 2),
            ],
            outputs_info=[
                TT.zeros((N, self.hidden_state_dim)),
                TT.zeros((N, self.action_space.n)),
            ]
        )[0]

        all_action_prob = all_action_prob.dimshuffle(1, 0, 2)

        return dict(
            action_prob=all_action_prob,
        )

        # bottleneck_mean_var, bottleneck_log_std_var = L.get_output(
        #     [self.l_bottleneck_mean, self.l_bottleneck_log_std],
        #     inputs={self.l_raw_obs: obs_var, self.l_prev_hidden: prev_hidden_var}
        # )
        # bottleneck_var = bottleneck_epsilon * TT.exp(bottleneck_log_std_var) + bottleneck_mean_var
        # obs_var = bottleneck_var
        #
        #
        # hidden_mean_var, hidden_log_std_var = L.get_output(
        #     [self.l_hidden_mean, self.l_hidden_log_std],
        #     inputs={self.l_obs: obs_var, self.l_prev_hidden: prev_hidden_var}
        # )
        # hidden_var = hidden_epsilon * TT.exp(hidden_log_std_var) + hidden_mean_var
        #
        # action_prob_var = L.get_output(self.l_action_prob, inputs={
        #     self.l_obs: obs_var,
        #     self.l_hidden: hidden_var,
        # })
        # ret = dict(
        #     action_prob=action_prob_var,
        #     hidden_mean=hidden_mean_var,
        #     hidden_log_std=hidden_log_std_var,
        #     bottleneck_mean_var=bottleneck_mean_var,
        #     bottleneck_log_std_var=bottleneck_log_std_var,
        # )
        # return ret

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        # hidden_kl = self.hidden_dist.kl_sym(
        #     dict(prob=old_dist_info_vars["hidden_prob"]),
        #     dict(prob=new_dist_info_vars["hidden_prob"])
        # )
        # only take hidden kl into account if switching goal
        # if self.use_decision_nodes:
        #     cond = old_dist_info_vars["switch_goal"][:, 0]
        #     hidden_kl = hidden_kl * cond + TT.zeros([TT.shape(hidden_kl)[0]]) * (1 - cond)
        return self.action_dist.kl_sym(
            dict(prob=old_dist_info_vars["action_prob"]),
            dict(prob=new_dist_info_vars["action_prob"])
        )
        # ret = hidden_kl + action_kl
        # if self.use_decision_nodes:
        #     decision_kl = self.decision_dist.kl_sym(
        #         dict(p=old_dist_info_vars["decision_prob"]),
        #         dict(p=new_dist_info_vars["decision_prob"]),
        #     )
        #     ret += decision_kl
        # if self.use_bottleneck:
        #     ret += self.bottleneck_dist.kl_sym(
        #         dict(mean=old_dist_info_vars["bottleneck_mean"], log_std=old_dist_info_vars["bottleneck_log_std"]),
        #         dict(mean=new_dist_info_vars["bottleneck_mean"], log_std=new_dist_info_vars["bottleneck_log_std"]),
        #     )
        # return ret

    def likelihood_ratio_sym(self, action_var, old_dist_info_vars, new_dist_info_vars):
        # hidden_lr = self.hidden_dist.likelihood_ratio_sym(
        #     TT.cast(old_dist_info_vars["hidden_state"], 'uint8'),
        #     dict(prob=old_dist_info_vars["hidden_prob"]),
        #     dict(prob=new_dist_info_vars["hidden_prob"])
        # )
        # # only take hidden kl into account if switching goal
        # if self.use_decision_nodes:
        #     cond = old_dist_info_vars["switch_goal"][:, 0]
        #     hidden_lr = hidden_lr * cond + TT.ones([TT.shape(hidden_lr)[0]]) * (1 - cond)
        return self.action_dist.likelihood_ratio_sym(
            action_var,
            dict(prob=old_dist_info_vars["action_prob"]),
            dict(prob=new_dist_info_vars["action_prob"])
        )
        # ret = hidden_lr * action_lr
        # if self.use_decision_nodes:
        #     decision_lr = self.decision_dist.likelihood_ratio_sym(
        #         TT.cast(old_dist_info_vars["switch_goal"], 'uint8'),
        #         dict(p=old_dist_info_vars["decision_prob"]),
        #         dict(p=new_dist_info_vars["decision_prob"]),
        #     )
        #     ret *= decision_lr
        # # we don't need to compute the term for the bottleneck, since it will be handled by reparameterization
        # return ret

    def log_likelihood_sym(self, action_var, dist_info_vars):
        # hidden_logli = self.hidden_dist.log_likelihood_sym(
        #     TT.cast(dist_info_vars["hidden_state"], 'uint8'),
        #     dict(prob=dist_info_vars["hidden_prob"]),
        # )
        # # only take hidden kl into account if switching goal
        # if self.use_decision_nodes:
        #     cond = dist_info_vars["switch_goal"][:, 0]
        #     hidden_logli = hidden_logli * cond + TT.zeros([TT.shape(hidden_logli)[0]]) * (1 - cond)
        return self.action_dist.log_likelihood_sym(
            action_var,
            dict(prob=dist_info_vars["action_prob"]),
        )
        # ret = hidden_logli + action_logli
        # if self.use_decision_nodes:
        #     decision_logli = self.decision_dist.log_likelihood_sym(
        #         TT.cast(dist_info_vars["switch_goal"], 'uint8'),
        #         dict(p=dist_info_vars["decision_prob"]),
        #     )
        #     ret += decision_logli
        # # we don't need to compute the term for the bottleneck, since it will be handled by reparameterization
        # return ret

    def entropy(self, dist_info):
        # the entropy is a bit difficult to estimate
        # for now we'll keep things simple and compute H(a|s,h)
        return self.action_dist.entropy(dict(prob=dist_info["action_prob"]))

    def reset(self):
        # if self.random_reset:
        #     self.hidden_state = np.eye(self.n_subgoals)[np.random.randint(low=0, high=self.n_subgoals)]
        # else:
        # always start on a fixed hidden state
        self.hidden_state = np.zeros((self.hidden_state_dim,))

    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        prev_hidden = self.hidden_state
        bottleneck_mean, bottleneck_log_std = [x[0] for x in self.f_bottleneck_dist([flat_obs], [prev_hidden])]
        bottleneck_epsilon = np.random.standard_normal((self.bottleneck_dim,))
        bottleneck = bottleneck_epsilon * np.exp(bottleneck_log_std) + bottleneck_mean
        obs = bottleneck
        hidden_mean, hidden_log_std = [x[0] for x in self.f_hidden_dist([obs], [prev_hidden])]
        hidden_epsilon = np.random.standard_normal((self.hidden_state_dim,))
        hidden = hidden_epsilon * np.exp(hidden_log_std) + hidden_mean
        action_prob = self.f_action_prob([obs], [hidden])[0]
        action = special.weighted_sample(action_prob, np.arange(self.action_space.n))
        agent_info = dict(
            # bottleneck_mean=bottleneck_mean,
            # bottleneck_log_std=bottleneck_log_std,
            bottleneck_epsilon=bottleneck_epsilon,
            # hidden_mean=hidden_mean,
            # hidden_log_std=hidden_log_std,
            hidden_epsilon=hidden_epsilon,
            action_prob=action_prob,
            # prev_hidden=prev_hidden,
        )
        return action, agent_info
