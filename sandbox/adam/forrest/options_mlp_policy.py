import numpy as np
import theano.tensor as TT

from rllab.core.lasagne_powered import LasagnePowered
from rllab.spaces import Box

from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from rllab.spaces.discrete import Discrete
from rllab.envs.env_spec import EnvSpec

from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from sandbox.adam.forrest.hierarchical import Hierarchical
from rllab.misc.overrides import overrides
from rllab.misc import logger


class OptionsMLPPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            num_options,
            hidden_sizes=((32,), (32, 32), (32,),),
            dist_cls=Hierarchical
    ):
        Serializable.quick_init(self, locals())
        self._action_dim = env_spec.action_space.flat_dim

        termination_action_space = Discrete(2)  # binary termination events
        option_action_space = Discrete(num_options)
        termination_observation_space = env_spec.observation_space
        option_observation_space = env_spec.observation_space

        self.option_policy = CategoricalMLPPolicy(env_spec=EnvSpec(option_observation_space, option_action_space),
                                                  hidden_sizes=hidden_sizes[0])

        self.action_policies = \
            [GaussianMLPPolicy(env_spec=env_spec, hidden_sizes=hidden_sizes[1]) for _ in range(0, num_options)]

        self.termination_policies = \
            [CategoricalMLPPolicy(
             env_spec=EnvSpec(termination_observation_space, termination_action_space),
             hidden_sizes=hidden_sizes[2]) for _ in range(0, num_options)]

        self._num_options = num_options
        self._dist = dist_cls(num_options, self._action_dim)
        self._prev_option = np.float32(0)
        self.reset()
        output_layers = [layer for layer in self.option_policy.output_layers] + \
                        [layer for o in range(0, num_options) for layer in self.action_policies[o].output_layers] + \
                        [layer for o in range(0, num_options) for layer in self.action_policies[o].output_layers]

        LasagnePowered.__init__(self, output_layers)
        assert isinstance(env_spec.action_space, Box)

    def reset(self):
        self._prev_option = 0

    def dist_info_sym(self, obs_var, state_info_vars):
        dist_info = {}
        option_info = self.option_policy.dist_info_sym(obs_var)
        term_infos = [self.termination_policies[o].dist_info_sym(obs_var)['prob'] for o in range(0, self._num_options)]
        for o in range(self._num_options):
            a_info = self.action_policies[o].dist_info_sym(obs_var)
            dist_info['action_mean_%s' % o] = a_info['mean']
            dist_info['action_log_std_%s' % o] = a_info['log_std']
            dist_info['termination_prob_%s' % o] = term_infos[o]

        term_probs = TT.as_tensor_variable(term_infos).dimshuffle(1, 0, 2)
        term_prob = TT.sum(state_info_vars['prev_option'].dimshuffle(0, 1, 'x') * term_probs, axis=1)

        dist_info['markov_prob'] = term_prob[:, 1].dimshuffle(0, 'x') * option_info['prob'] + \
             term_prob[:, 0].dimshuffle(0, 'x') * state_info_vars['prev_option']

        dist_info['markov_prob'] = dist_info['markov_prob'] / TT.sum(dist_info['markov_prob'], axis=1).dimshuffle(0, 'x')
        dist_info['option_prob'] = option_info['prob']
        dist_info['message_prob'] = state_info_vars['message_prob']
        dist_info['prev_option'] = state_info_vars['prev_option']
        return dist_info

    @overrides
    def get_action(self, observation):
        # TODO reduce redundant computation
        action_info = {}
        prev_option_one_hot = np.zeros(self._num_options)

        prev_option_one_hot[self._prev_option] = 1.0
        # prev_option_one_hot[0] = 1.0
        action_info['prev_option'] = np.asarray(prev_option_one_hot)

        # Sample action
        # termination, termination_info = self.termination_policies[self._prev_option].get_action(observation)
        # action_info['termination_prob'] = termination_info['prob']

        actions = []
        action_infos = []
        termination_infos = []

        sampled_option, option_info = self.option_policy.get_action(observation)

        markov_prob = np.zeros(self._num_options)
        # Store dist_info
        for o in range(0, self._num_options):
            a, a_info = self.action_policies[o].get_action(observation)
            termination, t_info = self.termination_policies[o].get_action(observation)
            actions.append(a)
            action_infos.append(a_info)
            termination_infos.append(t_info)
            actions.append(a)

            # markov_prob[o] = t_info['prob'][1] * option_info['prob'][o]
            # markov_prob[o] = termination_infos[0]['prob'][1] * option_info['prob'][o]
            # termination_infos[0]['prob'][1]
            if o == self._prev_option:
                markov_prob = (t_info['prob'][1] * option_info['prob'])
                markov_prob[o] += t_info['prob'][0]
                # markov_prob[o] += t_info['prob'][0]
                if termination:
                    option = sampled_option
                else:
                    option = self._prev_option
            # else:
            #     markov_prob[o] = (t_info['prob'][1] * option_info['prob'][o])
            action_info['action_mean_%s' % o] = a_info['mean']
            action_info['action_log_std_%s' % o] = a_info['log_std']
            action_info['termination_prob_%s' % o] = t_info['prob']
        markov_prob /= np.sum(markov_prob)
        action_info['markov_prob'] = np.asarray(markov_prob)
        # action_info['markov_prob'] = termination_infos[0]['prob'][1] * option_info['prob']
        # action_info['markov_prob'] = t_info['prob'][1] * option_info['prob']
        action = actions[option]
        action_info['option_prob'] = option_info['prob']

        # Compute message probabilities, p(a, b, o | s, o-) for each b, o, o-
        # (Could absorb section above into loops below for slight speedup, but for now
        # keeping separate for clarity. Also can look into vectorizing loops below if too slow.)
        message_prob = np.zeros((self._num_options, 2, self._num_options))
        for o_prev in range(0, self._num_options):
            term_info = termination_infos[o_prev]
            for b in range(0, 2):
                if b:
                    o_info = option_info
                else:
                    densities = np.zeros(self._num_options)
                    densities[o_prev] = 1.0
                    o_info = dict(prob=np.asarray(densities))
                for o in range(0, self._num_options):
                    a_info = action_infos[o]
                    std = np.exp(a_info['log_std'])
                    mean = a_info['mean']
                    val = (action - mean) / std
                    p_a = 1.0 / np.sqrt(np.power(2.0*np.pi, mean.shape[-1]) *
                                        np.square(np.prod(std, axis=-1))) * \
                        np.exp(-0.5 * np.sum(np.square(val), axis=-1))
                    message_prob[o_prev, b, o] = term_info['prob'][b] * o_info['prob'][o] * p_a
        action_info['message_prob'] = message_prob
        self._prev_option = option
        return action, action_info

    def log_diagnostics(self, paths):
        # TODO implement correctly
        options_dist = np.vstack([path["agent_infos"]["prev_option"] for path in paths])
        logger.record_tabular('MeanOptionsDistribution', np.mean(options_dist, axis=0))
        last_options = paths[0]["agent_infos"]["prev_option"]
        for o in range(self._num_options):
            logger.record_tabular('OptionsDistribution%s' % o, last_options[:, o])

    @property
    def distribution(self):
        return self._dist

    @property
    def state_info_keys(self):
        return [("prev_option", 1), ("message_prob", 3)]
