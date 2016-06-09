from __future__ import print_function
from __future__ import absolute_import

from rllab.policies.base import StochasticPolicy
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.spaces.discrete import Discrete
from rllab.distributions.categorical import Categorical
from rllab.distributions.bernoulli import Bernoulli
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
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
                 use_decision_nodes=False,
                 use_bottleneck=False,
                 separate_bottlenecks=False,
                 deterministic_bottleneck=False,
                 bottleneck_nonlinear=False,
                 bottleneck_dim=5,
                 action_bottleneck_dim=None,
                 hidden_bottleneck_dim=None,
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
        :param use_bottleneck: whether to evaluate the mutual information terms using a bottleneck state
        :param separate_bottlenecks: whether to use different bottleneck networks for the hidden network and action
        network
        :param deterministic_bottleneck: whether to use deterministic bottleneck or stochastic bottleneck
        :param bottleneck_nonlinear: whether there should be nonlinearity applied to the bottleneck
        :param bottleneck_dim: dimension of bottleneck
        :param action_bottleneck_dim: dimension of action bottleneck. only used if separate_bottlenecks is True
        :param hidden_bottleneck_dim: dimension of hidden bottleneck. only used if separate_bottlenecks is True
        :param random_reset: whether to randomly set the first subgoal
        """
        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Discrete)
        # assert not use_decision_nodes
        assert not random_reset
        assert use_bottleneck
        assert deterministic_bottleneck
        assert bottleneck_nonlinear

        if not separate_bottlenecks:
            action_bottleneck_dim = bottleneck_dim
            hidden_bottleneck_dim = bottleneck_dim
        else:
            assert action_bottleneck_dim is not None
            assert hidden_bottleneck_dim is not None

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
        self.separate_bottlenecks = separate_bottlenecks
        self.action_bottleneck_dim = action_bottleneck_dim
        self.hidden_bottleneck_dim = hidden_bottleneck_dim
        self.deterministic_bottleneck = deterministic_bottleneck
        self.bottleneck_nonlinear = bottleneck_nonlinear

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

        action_bottleneck_network = MLP(
            input_layer=L.concat([l_raw_obs, l_prev_hidden], name="action_bottleneck_network_input"),
            hidden_sizes=bottleneck_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=hidden_nonlinearity,
            output_dim=action_bottleneck_dim,
            name="action_bottleneck_network"
        )

        if separate_bottlenecks:
            hidden_bottleneck_network = MLP(
                input_layer=L.concat([l_raw_obs, l_prev_hidden], name="hidden_bottleneck_network_input"),
                hidden_sizes=bottleneck_hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=hidden_nonlinearity,
                output_dim=hidden_bottleneck_dim,
                name="hidden_bottleneck_network"
            )
        else:
            hidden_bottleneck_network = action_bottleneck_network

        l_action_bottleneck_out = action_bottleneck_network.output_layer
        l_hidden_bottleneck_out = hidden_bottleneck_network.output_layer

        l_action_bottleneck = L.InputLayer(
            shape=(None, action_bottleneck_dim),
            name="action_bottleneck"
        )

        l_hidden_bottleneck = L.InputLayer(
            shape=(None, hidden_bottleneck_dim),
            name="hidden_bottleneck"
        )

        if self.use_bottleneck:
            l_action_obs = l_action_bottleneck
            l_hidden_obs = l_hidden_bottleneck
        else:
            l_action_obs = l_raw_obs
            l_hidden_obs = l_raw_obs

        decision_network = MLP(
            input_layer=L.concat([l_hidden_obs, l_prev_hidden], name="decision_network_input"),
            hidden_sizes=decision_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=TT.nnet.sigmoid,  # tf.nn.sigmoid,
            output_dim=1,
            name="decision_network",
        )
        hidden_network = MLP(
            input_layer=L.concat([l_hidden_obs, l_prev_hidden], name="hidden_network_input"),
            hidden_sizes=hid_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=TT.nnet.softmax,  # tf.nn.softmax,
            output_dim=n_subgoals,
            name="hidden_network"
        )
        action_network = MLP(
            input_layer=L.concat([l_action_obs, l_hidden], name="action_network_input"),
            hidden_sizes=action_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=TT.nnet.softmax,  # tf.nn.softmax,
            output_dim=env_spec.action_space.n,
            name="action_network"
        )

        l_decision_prob = decision_network.output_layer
        l_hidden_prob = hidden_network.output_layer
        l_action_prob = action_network.output_layer

        StochasticPolicy.__init__(self, env_spec=env_spec)

        used_layers = [l_hidden_prob, l_action_prob]
        if use_decision_nodes:
            used_layers += [l_decision_prob]
        if use_bottleneck:
            used_layers += [l_action_bottleneck_out]
            if separate_bottlenecks:
                used_layers += [l_hidden_bottleneck_out]
        LasagnePowered.__init__(self, used_layers)

        self.hid_hidden_sizes = hid_hidden_sizes
        self.action_hidden_sizes = action_hidden_sizes
        self.bottleneck_hidden_sizes = bottleneck_hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity

        self.l_hidden_prob = l_hidden_prob
        self.l_decision_prob = l_decision_prob
        self.l_action_prob = l_action_prob
        self.l_raw_obs = l_raw_obs
        self.l_action_obs = l_action_obs
        self.l_hidden_obs = l_hidden_obs
        self.l_prev_hidden = l_prev_hidden
        self.l_action_bottleneck = l_action_bottleneck
        self.l_hidden_bottleneck = l_hidden_bottleneck
        self.l_action_bottleneck_out = l_action_bottleneck_out
        self.l_hidden_bottleneck_out = l_hidden_bottleneck_out
        self.l_hidden = l_hidden

        self.hidden_dist = Categorical(self.n_subgoals)
        self.action_dist = Categorical(self.action_space.n)
        self.decision_dist = Bernoulli(1)
        # self.bottleneck_dist = DiagonalGaussian(self.bottleneck_dim)

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2,
                dtype=theano.config.floatX
            ) for k in self.state_info_keys
            }
        state_info_vars_list = [state_info_vars[k] for k in self.state_info_keys]

        self.f_dist_info = ext.compile_function(
            [l_raw_obs.input_var] + state_info_vars_list,
            self.dist_info_sym(l_raw_obs.input_var, state_info_vars)
        )

        hidden_prob_var = self.hidden_prob_sym(hidden_obs_var=l_hidden_obs.input_var,
                                               prev_hidden_var=l_prev_hidden.input_var)

        self.f_hidden_prob = ext.compile_function(
            [l_hidden_obs.input_var, l_prev_hidden.input_var],
            hidden_prob_var,
        )

        action_prob_var = self.action_prob_sym(action_obs_var=l_action_obs.input_var,
                                               hidden_var=l_hidden.input_var)

        self.f_action_prob = ext.compile_function(
            [l_action_obs.input_var, l_hidden.input_var],
            action_prob_var,
        )
        self.f_bottlenecks = ext.compile_function(
            [l_raw_obs.input_var, l_prev_hidden.input_var],
            L.get_output([l_action_bottleneck_out, l_hidden_bottleneck_out]),
        )

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
            ("hidden_state", (self.n_subgoals,)),
            ("prev_hidden", (self.n_subgoals,))
        ]
        return specs

    @property
    def dist_info_specs(self):
        specs = [
            ("action_prob", (self.action_space.n,)),
            ("hidden_prob", (self.n_subgoals,)),
            ("hidden_state", (self.n_subgoals,)),
        ]
        if self.use_bottleneck:
            specs += [
                ("action_bottleneck", (self.action_bottleneck_dim,)),
                ("hidden_bottleneck", (self.hidden_bottleneck_dim,)),
            ]
        return specs

    def dist_info(self, obs, state_infos):
        state_info_list = [state_infos[k] for k in self.state_info_keys]
        return self.f_dist_info(obs, *state_info_list)

    def hidden_prob_sym(self, hidden_obs_var, prev_hidden_var):
        hidden_prob_var = L.get_output(self.l_hidden_prob, inputs={
            self.l_hidden_obs: hidden_obs_var,
            self.l_prev_hidden: prev_hidden_var,
        })
        if self.use_decision_nodes:
            # the decision node is subsumed in reparametrizing the hidden probabilities
            decision_prob_var = L.get_output(self.l_decision_prob, inputs={
                self.l_hidden_obs: hidden_obs_var,
                self.l_prev_hidden: prev_hidden_var,
            })
            switch_prob_var = TT.tile(decision_prob_var, [1, self.n_subgoals])
            hidden_prob_var = switch_prob_var * hidden_prob_var + (1 - switch_prob_var) * prev_hidden_var
        return hidden_prob_var

    def action_prob_sym(self, action_obs_var, hidden_var):
        action_prob_var = L.get_output(self.l_action_prob, inputs={
            self.l_action_obs: action_obs_var,
            self.l_hidden: hidden_var,
        })
        return action_prob_var

    def dist_info_sym(self, obs_var, state_info_vars):
        prev_hidden_var = state_info_vars["prev_hidden"]
        hidden_var = state_info_vars["hidden_state"]

        if self.use_bottleneck:
            action_bottleneck_var, hidden_bottleneck_var = L.get_output(
                [self.l_action_bottleneck_out, self.l_hidden_bottleneck_out],
                inputs={self.l_raw_obs: obs_var, self.l_prev_hidden: prev_hidden_var}
            )
            action_obs = action_bottleneck_var
            hidden_obs = hidden_bottleneck_var
        else:
            action_obs = obs_var
            hidden_obs = obs_var
            action_bottleneck_var = None
            hidden_bottleneck_var = None

        hidden_prob_var = self.hidden_prob_sym(hidden_obs_var=hidden_obs, prev_hidden_var=prev_hidden_var)
        action_prob_var = self.action_prob_sym(action_obs_var=action_obs, hidden_var=hidden_var)

        ret = dict(
            hidden_prob=hidden_prob_var,
            action_prob=action_prob_var,
            hidden_state=hidden_var,
            action_bottleneck=action_bottleneck_var,
            hidden_bottleneck=hidden_bottleneck_var,
        )
        return ret

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        hidden_kl = self.hidden_dist.kl_sym(
            dict(prob=old_dist_info_vars["hidden_prob"]),
            dict(prob=new_dist_info_vars["hidden_prob"])
        )
        action_kl = self.action_dist.kl_sym(
            dict(prob=old_dist_info_vars["action_prob"]),
            dict(prob=new_dist_info_vars["action_prob"])
        )
        ret = hidden_kl + action_kl
        return ret

    def likelihood_ratio_sym(self, action_var, old_dist_info_vars, new_dist_info_vars):
        hidden_lr = self.hidden_dist.likelihood_ratio_sym(
            TT.cast(old_dist_info_vars["hidden_state"], 'uint8'),
            dict(prob=old_dist_info_vars["hidden_prob"]),
            dict(prob=new_dist_info_vars["hidden_prob"])
        )
        action_lr = self.action_dist.likelihood_ratio_sym(
            action_var,
            dict(prob=old_dist_info_vars["action_prob"]),
            dict(prob=new_dist_info_vars["action_prob"])
        )
        ret = hidden_lr * action_lr
        return ret

    def log_likelihood_sym(self, action_var, dist_info_vars):
        hidden_logli = self.hidden_dist.log_likelihood_sym(
            TT.cast(dist_info_vars["hidden_state"], 'uint8'),
            dict(prob=dist_info_vars["hidden_prob"]),
        )
        action_logli = self.action_dist.log_likelihood_sym(
            action_var,
            dict(prob=dist_info_vars["action_prob"]),
        )
        ret = hidden_logli + action_logli
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
        agent_info = dict()
        if self.use_bottleneck:
            action_bottleneck, hidden_bottleneck = [x[0] for x in self.f_bottlenecks([flat_obs], [prev_hidden])]
            action_obs = action_bottleneck
            hidden_obs = hidden_bottleneck
            agent_info["action_bottleneck"] = action_bottleneck
            agent_info["hidden_bottleneck"] = hidden_bottleneck
        else:
            action_obs = flat_obs
            hidden_obs = flat_obs

        hidden_prob = self.f_hidden_prob([hidden_obs], [prev_hidden])[0]

        self.hidden_state = np.eye(self.n_subgoals)[
            special.weighted_sample(hidden_prob, np.arange(self.n_subgoals))
        ]

        action_prob = self.f_action_prob([action_obs], [self.hidden_state])[0]
        action = special.weighted_sample(action_prob, np.arange(self.action_space.n))
        agent_info["action_prob"] = action_prob
        agent_info["hidden_prob"] = hidden_prob
        agent_info["hidden_state"] = self.hidden_state
        agent_info["prev_hidden"] = prev_hidden
        return action, agent_info

    def self_analyze(self):
        pass
