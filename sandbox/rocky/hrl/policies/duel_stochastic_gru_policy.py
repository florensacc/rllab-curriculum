


from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
from rllab.policies.base import StochasticPolicy
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import special
from rllab.core.network import MLP
import lasagne.layers as L
import theano.tensor as TT
import numpy as np


class DuelStochasticGRUPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(self, env_spec, master_policy):
        """
        :type master_policy: StochasticGRUPolicy
        """
        Serializable.quick_init(self, locals())
        self.master_policy = master_policy

        assert not master_policy.use_bottleneck

        hidden_network = MLP(
            input_layer=L.concat([master_policy.l_hidden_obs, master_policy.l_prev_hidden],
                                 name="hidden_network_input"),
            hidden_sizes=master_policy.hid_hidden_sizes,
            hidden_nonlinearity=master_policy.hidden_nonlinearity,
            output_nonlinearity=TT.nnet.softmax,
            output_dim=master_policy.n_subgoals,
            name="hidden_network"
        )

        l_hidden_prob = hidden_network.output_layer

        used_layers = [l_hidden_prob, master_policy.l_action_prob]
        if master_policy.use_decision_nodes:
            used_layers += [master_policy.l_decision_prob]

        self.hidden_state = None

        self.l_hidden_prob = l_hidden_prob

        StochasticPolicy.__init__(self, env_spec)
        LasagnePowered.__init__(self, used_layers)

        hidden_prob_var = self.hidden_prob_sym(hidden_obs_var=master_policy.l_hidden_obs.input_var,
                                               prev_hidden_var=master_policy.l_prev_hidden.input_var)

        self.f_hidden_prob = ext.compile_function(
            [master_policy.l_hidden_obs.input_var, master_policy.l_prev_hidden.input_var],
            hidden_prob_var,
        )

    @property
    def random_reset(self):
        return self.master_policy.random_reset

    @property
    def use_bottleneck(self):
        return self.master_policy.use_bottleneck

    @property
    def n_subgoals(self):
        return self.master_policy.n_subgoals

    @property
    def distribution(self):
        return self.master_policy.distribution

    @property
    def state_info_keys(self):
        return self.master_policy.state_info_keys

    def action_prob_sym(self, obs_var, hidden_var):
        return self.master_policy.action_prob_sym(obs_var, hidden_var)

    @property
    def action_dist(self):
        return self.master_policy.action_dist

    @property
    def hidden_dist(self):
        return self.master_policy.hidden_dist

    @property
    def f_action_prob(self):
        return self.master_policy.f_action_prob

    def hidden_prob_sym(self, hidden_obs_var, prev_hidden_var):
        hidden_prob_var = L.get_output(self.l_hidden_prob, inputs={
            self.master_policy.l_hidden_obs: hidden_obs_var,
            self.master_policy.l_prev_hidden: prev_hidden_var,
        })
        if self.master_policy.use_decision_nodes:
            # the decision node is subsumed in reparametrizing the hidden probabilities
            decision_prob_var = L.get_output(self.master_policy.l_decision_prob, inputs={
                self.master_policy.l_hidden_obs: hidden_obs_var,
                self.master_policy.l_prev_hidden: prev_hidden_var,
            })
            switch_prob_var = TT.tile(decision_prob_var, [1, self.n_subgoals])
            hidden_prob_var = switch_prob_var * hidden_prob_var + (1 - switch_prob_var) * prev_hidden_var
        return hidden_prob_var

    def dist_info_sym(self, obs_var, state_info_vars):
        prev_hidden_var = state_info_vars["prev_hidden"]
        hidden_var = state_info_vars["hidden_state"]

        hidden_prob_var = self.hidden_prob_sym(hidden_obs_var=obs_var, prev_hidden_var=prev_hidden_var)
        action_prob_var = self.master_policy.action_prob_sym(action_obs_var=obs_var, hidden_var=hidden_var)

        ret = dict(
            hidden_prob=hidden_prob_var,
            action_prob=action_prob_var,
            hidden_state=hidden_var,
        )
        return ret

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

        hidden_prob = self.f_hidden_prob([flat_obs], [prev_hidden])[0]

        self.hidden_state = np.eye(self.n_subgoals)[
            special.weighted_sample(hidden_prob, np.arange(self.n_subgoals))
        ]

        action_prob = self.master_policy.f_action_prob([flat_obs], [self.hidden_state])[0]
        action = special.weighted_sample(action_prob, np.arange(self.action_space.n))
        agent_info["action_prob"] = action_prob
        agent_info["hidden_prob"] = hidden_prob
        agent_info["hidden_state"] = self.hidden_state
        agent_info["prev_hidden"] = prev_hidden
        return action, agent_info
