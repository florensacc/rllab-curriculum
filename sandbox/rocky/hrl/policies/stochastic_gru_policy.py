from __future__ import print_function
from __future__ import absolute_import

from rllab.policies.base import StochasticPolicy
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.spaces.discrete import Discrete
from rllab.distributions.categorical import Categorical
from sandbox.rocky.hrl.distributions.bernoulli import Bernoulli
from rllab.envs.base import EnvSpec
from rllab.misc import ext, special
import lasagne.layers as L
import numpy as np
import theano
import theano.tensor as TT
import lasagne.nonlinearities as NL
import lasagne.init as LI

floatX = theano.config.floatX


class MLP(object):
    def __init__(self, output_dim, hidden_sizes, hidden_nonlinearity, output_nonlinearity,
                 hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
                 output_W_init=LI.GlorotUniform(), output_b_init=LI.Constant(0.),
                 name=None, input_var=None, input_layer=None, input_shape=None):

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        if input_layer is None:
            assert input_shape is not None, "must provide either input_shape or input_layer"
            l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var)
        else:
            l_in = input_layer
        self._layers = [l_in]
        l_hid = l_in
        for idx, hidden_size in enumerate(hidden_sizes):
            l_hid = L.DenseLayer(
                l_hid,
                num_units=hidden_size,
                nonlinearity=hidden_nonlinearity,
                name="%shidden_%d" % (prefix, idx),
                W=hidden_W_init,
                b=hidden_b_init,
            )
            self._layers.append(l_hid)
        l_out = L.DenseLayer(
            l_hid,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
            name="%soutput" % (prefix,),
            W=output_W_init,
            b=output_b_init,
        )
        self._layers.append(l_out)
        self._l_in = l_in
        self._l_out = l_out
        if isinstance(l_in, L.InputLayer):
            self._input_var = l_in.input_var
        else:
            self._input_var = None
        self._output = L.get_output(l_out)

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output


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

    def __init__(self, env_spec, n_subgoals, hidden_sizes=(32, 32), use_decision_nodes=True,
                 hid_hidden_sizes=None, decision_hidden_sizes=None, action_hidden_sizes=None,
                 hidden_nonlinearity=NL.tanh):
        """
        :type env_spec: EnvSpec
        :param use_decision_nodes: whether to have decision units, which governs whether the subgoals should be
        resampled
        """
        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Discrete)

        if hid_hidden_sizes is None:
            hid_hidden_sizes = hidden_sizes
        if decision_hidden_sizes is None:
            decision_hidden_sizes = hidden_sizes
        if action_hidden_sizes is None:
            action_hidden_sizes = hidden_sizes

        l_prev_hidden = L.InputLayer(
            shape=(None, n_subgoals),
            name="prev_hidden",
        )
        l_hidden = L.InputLayer(
            shape=(None, n_subgoals),
            name="hidden",
        )
        l_obs = L.InputLayer(
            shape=(None, env_spec.observation_space.flat_dim),
            name="obs",
        )

        decision_network = MLP(
            input_layer=L.concat([l_obs, l_prev_hidden]),
            hidden_sizes=decision_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=NL.sigmoid,
            output_dim=1,
        )
        hidden_network = MLP(
            input_layer=L.concat([l_obs, l_prev_hidden]),
            hidden_sizes=hid_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=NL.softmax,
            output_dim=n_subgoals,
        )
        action_network = MLP(
            input_layer=L.concat([l_obs, l_hidden]),
            hidden_sizes=action_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=NL.softmax,
            output_dim=env_spec.action_space.n,
        )

        l_decision_prob = decision_network.output_layer
        l_hidden_prob = hidden_network.output_layer
        l_action_prob = action_network.output_layer

        self.hidden_state = None
        # self.env_spec = env_spec
        self.n_subgoals = n_subgoals
        self.use_decision_nodes = use_decision_nodes

        self.f_hidden_prob = ext.compile_function(
            [l_obs.input_var, l_prev_hidden.input_var],
            L.get_output(l_hidden_prob),
        )
        self.f_decision_prob = ext.compile_function(
            [l_obs.input_var, l_prev_hidden.input_var],
            L.get_output(l_decision_prob).flatten(),
        )
        self.f_action_prob = ext.compile_function(
            [l_obs.input_var, l_hidden.input_var],
            L.get_output(l_action_prob),
        )

        StochasticPolicy.__init__(self, env_spec=env_spec)
        if self.use_decision_nodes:
            LasagnePowered.__init__(self, [l_hidden_prob, l_decision_prob, l_action_prob])
        else:
            LasagnePowered.__init__(self, [l_hidden_prob, l_action_prob])

        self.l_hidden_prob = l_hidden_prob
        self.l_decision_prob = l_decision_prob
        self.l_action_prob = l_action_prob
        self.l_obs = l_obs
        self.l_prev_hidden = l_prev_hidden
        self.l_hidden = l_hidden

        self.hidden_dist = Categorical(self.n_subgoals)
        self.action_dist = Categorical(self.action_space.n)
        self.decision_dist = Bernoulli(1)
        # for dist in hidden_spec:

    @property
    def distribution(self):
        return self

    @property
    def state_info_keys(self):
        keys = ["action_prob", "hidden_prob", "hidden_state", "prev_hidden"]
        if self.use_decision_nodes:
            keys += ["decision_prob", "switch_goal"]
        return keys

    @property
    def dist_info_keys(self):
        keys = ["action_prob", "hidden_prob", "hidden_state"]
        if self.use_decision_nodes:
            keys += ["decision_prob", "switch_goal"]
        return keys

    def dist_info_sym(self, obs_var, state_info_vars):
        prev_hidden_var = state_info_vars["prev_hidden"]
        hidden_var = state_info_vars["hidden_state"]

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
        if self.use_decision_nodes:
            switch_goal_var = state_info_vars["switch_goal"]
            decision_prob_var = L.get_output(self.l_decision_prob, inputs={
                self.l_obs: obs_var,
                self.l_prev_hidden: prev_hidden_var,
            })
            hidden_prob_var = TT.switch(
                TT.tile(switch_goal_var, [1, self.n_subgoals]),
                hidden_prob_var,
                prev_hidden_var
            )
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
            hidden_kl = TT.switch(
                old_dist_info_vars["switch_goal"][:, 0],
                hidden_kl,
                TT.zeros([hidden_kl.shape[0]]),
            )
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
            hidden_lr = TT.switch(
                old_dist_info_vars["switch_goal"][:, 0],
                hidden_lr,
                TT.ones([hidden_lr.shape[0]])
            )
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

    def entropy(self, dist_info):
        # the entropy is a bit difficult to estimate
        # for now we'll keep things simple and compute H(a|s,h)
        return self.action_dist.entropy(dict(prob=dist_info["action_prob"]))

    def reset(self):
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
        return action, agent_info
