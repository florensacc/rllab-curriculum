from __future__ import print_function
from __future__ import absolute_import
from rllab import config

from rllab.policies.base import StochasticPolicy
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.spaces.discrete import Discrete
from rllab.distributions.categorical import Categorical
from rllab.distributions.bernoulli import Bernoulli
from rllab.misc import tensor_utils
from rllab.envs.base import EnvSpec
from rllab.misc import ext
from rllab.misc import special
from rllab.core.network import MLP
import lasagne.layers as L
import theano.tensor as TT
import numpy as np


class StochasticGRUPolicy(StochasticPolicy, LasagnePowered, Serializable):
    """
    Structure the hierarchical policy as a recurrent network with stochastic gated recurrent unit, where the
    stochastic component of the hidden state will play the role of internal goals. Binary (or continuous) decision
    gates control the updates to the internal goals.

    There are too many design variables here:

    - Should the hidden states (subgoals) be deterministic or stochastic? If stochastic, should they be discrete or
    continuous, or both?
    - Same question for the decision gate.
    - Should we have separate decision gates for each hidden unit?

    For simplicity, we'll go with the following design choice:
    - discrete hidden state which is precisely one categorical variable, representing the subgoal
    - discrete decision nodes
    """

    def __init__(self,
                 env_spec,
                 n_subgoals,
                 hidden_sizes=(32, 32),
                 use_decision_nodes=True,
                 use_bottleneck=False,
                 bottleneck_dim=5,
                 hid_hidden_sizes=None,
                 decision_hidden_sizes=None,
                 action_hidden_sizes=None,
                 bottleneck_hidden_sizes=None,
                 random_reset=False,
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
        self.n_subgoals = n_subgoals
        self.use_decision_nodes = use_decision_nodes
        self.random_reset = random_reset
        self.use_bottleneck = use_bottleneck
        self.bottleneck_dim = bottleneck_dim

        l_prev_hidden = L.InputLayer(
            shape=(None, n_subgoals),
            name="prev_hidden",
        )
        l_hidden = L.InputLayer(
            shape=(None, n_subgoals),
            name="hidden",
        )
        l_raw_obs = L.InputLayer(
            shape=(None, env_spec.observation_space.flat_dim),
            name="obs",
        )

        bottleneck_network = MLP(
            input_layer=l_raw_obs,
            hidden_sizes=bottleneck_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=None,
            output_dim=bottleneck_dim,
            name="bottleneck_network"
        )

        l_bottleneck = bottleneck_network.output_layer

        if self.use_bottleneck:
            l_obs = l_bottleneck
        else:
            l_obs = l_raw_obs

        decision_network = MLP(
            input_layer=L.concat([l_obs, l_prev_hidden], name="decision_network_input"),
            hidden_sizes=decision_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=TT.nnet.sigmoid,  # tf.nn.sigmoid,
            output_dim=1,
            name="decision_network",
        )
        hidden_network = MLP(
            input_layer=L.concat([l_obs, l_prev_hidden], name="hidden_network_input"),
            hidden_sizes=hid_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=TT.nnet.softmax,  # tf.nn.softmax,
            output_dim=n_subgoals,
            name="hidden_network"
        )
        action_network = MLP(
            input_layer=L.concat([l_obs, l_hidden], name="action_network_input"),
            hidden_sizes=action_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=TT.nnet.softmax,  # tf.nn.softmax,
            output_dim=env_spec.action_space.n,
            name="action_network"
        )

        l_decision_prob = decision_network.output_layer
        l_hidden_prob = hidden_network.output_layer
        l_action_prob = action_network.output_layer

        self.f_hidden_prob = ext.compile_function(
            [l_raw_obs.input_var, l_prev_hidden.input_var],
            L.get_output(l_hidden_prob),
        )
        self.f_decision_prob = ext.compile_function(
            [l_raw_obs.input_var, l_prev_hidden.input_var],
            TT.reshape(L.get_output(l_decision_prob), [-1])
        )
        self.f_action_prob = ext.compile_function(
            [l_raw_obs.input_var, l_hidden.input_var],
            L.get_output(l_action_prob),
        )
        self.f_bottleneck = ext.compile_function(
            [l_raw_obs.input_var],
            L.get_output(l_bottleneck),
        )

        StochasticPolicy.__init__(self, env_spec=env_spec)
        if self.use_decision_nodes:
            LasagnePowered.__init__(self, [l_hidden_prob, l_decision_prob, l_action_prob])
        else:
            LasagnePowered.__init__(self, [l_hidden_prob, l_action_prob])

        self.l_hidden_prob = l_hidden_prob
        self.l_decision_prob = l_decision_prob
        self.l_action_prob = l_action_prob
        self.l_raw_obs = l_raw_obs
        self.l_obs = l_obs
        self.l_prev_hidden = l_prev_hidden
        self.l_bottleneck = l_bottleneck
        self.l_hidden = l_hidden

        self.hidden_dist = Categorical(self.n_subgoals)
        self.action_dist = Categorical(self.action_space.n)
        self.decision_dist = Bernoulli(1)

    @property
    def distribution(self):
        return self

    @property
    def dist_info_keys(self):
        return [k for k, _ in self.dist_info_specs]

    @property
    def state_info_keys(self):
        return [k for k, _ in self.state_info_specs]

    @property
    def state_info_specs(self):
        specs = [
            # ("action_prob", (self.action_space.n,)),
            # ("hidden_prob", (self.n_subgoals,)),
            ("hidden_state", (self.n_subgoals,)),
            ("prev_hidden", (self.n_subgoals,))
        ]
        if self.use_decision_nodes:
            specs += [
                ("decision_prob", (1,)),
                ("switch_goal", (1,)),
            ]
        return specs

    @property
    def dist_info_specs(self):
        specs = [
            ("action_prob", (self.action_space.n,)),
            ("hidden_prob", (self.n_subgoals,)),
            ("hidden_state", (self.n_subgoals,)),
        ]
        if self.use_decision_nodes:
            specs += [
                ("decision_prob", (1,)),
                ("switch_goal", (1,)),
            ]
        return specs

    def dist_info_sym(self, obs_var, state_info_vars):
        prev_hidden_var = state_info_vars["prev_hidden"]
        hidden_var = state_info_vars["hidden_state"]

        if self.use_bottleneck:
            obs_var = L.get_output(self.l_bottleneck, inputs={self.l_raw_obs: obs_var})
            bottleneck_var = obs_var
        else:
            bottleneck_var = None

        hidden_prob_var = L.get_output(self.l_hidden_prob, inputs={
            self.l_obs: obs_var,
            self.l_prev_hidden: prev_hidden_var,
        })
        # if not switching, hidden will be the same as before with probability 1

        action_prob_var = L.get_output(self.l_action_prob, inputs={
            self.l_obs: obs_var,
            self.l_hidden: hidden_var,
        })
        ret = dict(
            hidden_prob=hidden_prob_var,
            action_prob=action_prob_var,
            hidden_state=hidden_var,
        )
        if self.use_bottleneck:
            ret["bottleneck"] = bottleneck_var
        if self.use_decision_nodes:
            switch_goal_var = state_info_vars["switch_goal"]
            decision_prob_var = L.get_output(self.l_decision_prob, inputs={
                self.l_obs: obs_var,
                self.l_prev_hidden: prev_hidden_var,
            })
            cond = TT.tile(switch_goal_var, [1, self.n_subgoals])
            hidden_prob_var = hidden_prob_var * cond + prev_hidden_var * (1 - cond)
            ret["switch_goal"] = switch_goal_var
            ret["hidden_prob"] = hidden_prob_var
            ret["decision_prob"] = decision_prob_var
        return ret

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        hidden_kl = self.hidden_dist.kl_sym(
            dict(prob=old_dist_info_vars["hidden_prob"]),
            dict(prob=new_dist_info_vars["hidden_prob"])
        )
        # only take hidden kl into account if switching goal
        if self.use_decision_nodes:
            cond = old_dist_info_vars["switch_goal"][:, 0]
            hidden_kl = hidden_kl * cond + TT.zeros([TT.shape(hidden_kl)[0]]) * (1 - cond)
        action_kl = self.action_dist.kl_sym(
            dict(prob=old_dist_info_vars["action_prob"]),
            dict(prob=new_dist_info_vars["action_prob"])
        )
        ret = hidden_kl + action_kl
        if self.use_decision_nodes:
            decision_kl = self.decision_dist.kl_sym(
                dict(p=old_dist_info_vars["decision_prob"]),
                dict(p=new_dist_info_vars["decision_prob"]),
            )
            ret += decision_kl
        return ret

    def likelihood_ratio_sym(self, action_var, old_dist_info_vars, new_dist_info_vars):
        hidden_lr = self.hidden_dist.likelihood_ratio_sym(
            TT.cast(old_dist_info_vars["hidden_state"], 'uint8'),
            dict(prob=old_dist_info_vars["hidden_prob"]),
            dict(prob=new_dist_info_vars["hidden_prob"])
        )
        # only take hidden kl into account if switching goal
        if self.use_decision_nodes:
            cond = old_dist_info_vars["switch_goal"][:, 0]
            hidden_lr = hidden_lr * cond + TT.ones([TT.shape(hidden_lr)[0]]) * (1 - cond)
        action_lr = self.action_dist.likelihood_ratio_sym(
            action_var,
            dict(prob=old_dist_info_vars["action_prob"]),
            dict(prob=new_dist_info_vars["action_prob"])
        )
        ret = hidden_lr * action_lr
        if self.use_decision_nodes:
            decision_lr = self.decision_dist.likelihood_ratio_sym(
                TT.cast(old_dist_info_vars["switch_goal"], 'uint8'),
                dict(p=old_dist_info_vars["decision_prob"]),
                dict(p=new_dist_info_vars["decision_prob"]),
            )
            ret *= decision_lr
        return ret

    def log_likelihood_sym(self, action_var, dist_info_vars):
        hidden_logli = self.hidden_dist.log_likelihood_sym(
            TT.cast(dist_info_vars["hidden_state"], 'uint8'),
            dict(prob=dist_info_vars["hidden_prob"]),
        )
        # only take hidden kl into account if switching goal
        if self.use_decision_nodes:
            cond = dist_info_vars["switch_goal"][:, 0]
            hidden_logli = hidden_logli * cond + TT.zeros([TT.shape(hidden_logli)[0]]) * (1 - cond)
        action_logli = self.action_dist.log_likelihood_sym(
            action_var,
            dict(prob=dist_info_vars["action_prob"]),
        )
        ret = hidden_logli + action_logli
        if self.use_decision_nodes:
            decision_logli = self.decision_dist.log_likelihood_sym(
                TT.cast(dist_info_vars["switch_goal"], 'uint8'),
                dict(p=dist_info_vars["decision_prob"]),
            )
            ret += decision_logli
        return ret

    def entropy(self, dist_info):
        # the entropy is a bit difficult to estimate
        # for now we'll keep things simple and compute H(a|s,h)
        return self.action_dist.entropy(dict(prob=dist_info["action_prob"]))

    def reset(self):
        if self.random_reset:
            self.hidden_state = np.eye(self.n_subgoals)[np.random.randint(low=0, high=self.n_subgoals)]
        else:
            # always start on the first hidden state
            self.hidden_state = np.eye(self.n_subgoals)[0]

    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        prev_hidden = self.hidden_state
        if self.use_decision_nodes:
            decision_prob = self.f_decision_prob([flat_obs], [prev_hidden])[0]
            switch_goal = np.random.binomial(n=1, p=decision_prob)
        else:
            switch_goal = True
            decision_prob = None
        if switch_goal:
            hidden_prob = self.f_hidden_prob([flat_obs], [prev_hidden])[0]
            self.hidden_state = np.eye(self.n_subgoals)[
                special.weighted_sample(hidden_prob, np.arange(self.n_subgoals))
            ]
        else:
            hidden_prob = np.copy(self.hidden_state)
        action_prob = self.f_action_prob([flat_obs], [self.hidden_state])[0]
        action = special.weighted_sample(action_prob, np.arange(self.action_space.n))
        agent_info = dict(
            action_prob=action_prob,
            hidden_prob=hidden_prob,
            hidden_state=self.hidden_state,
            prev_hidden=prev_hidden,
        )
        if self.use_decision_nodes:
            agent_info["switch_goal"] = np.asarray([switch_goal])
            agent_info["decision_prob"] = np.asarray([decision_prob])
        if self.use_bottleneck:
            agent_info["bottleneck"] = self.f_bottleneck([flat_obs])[0]
        return action, agent_info
