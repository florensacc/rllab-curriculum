from rllab.algo.batch_polopt import BatchPolopt
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.special import to_onehot
from rllab.misc import categorical_dist
from rllab.policy.categorical_mlp_policy import CategoricalMLPPolicy
import numpy as np


class BatchHRL(BatchPolopt, Serializable):

    def __init__(
            self,
            high_algo,
            low_algo,
            mi_coeff=0.1,
            subgoal_interval=3,
            **kwargs):
        super(BatchHRL, self).__init__(**kwargs)
        Serializable.quick_init(self, locals())
        self._high_algo = high_algo
        self._low_algo = low_algo
        self._mi_coeff = mi_coeff
        self._subgoal_interval = subgoal_interval

    @property
    def high_algo(self):
        return self._high_algo

    @property
    def low_algo(self):
        return self._low_algo

    @overrides
    def init_opt(self, mdp_spec, policy, baseline, **kwargs):
        with logger.tabular_prefix('Hi_'), logger.prefix('Hi | '):
            high_opt_info = \
                self.high_algo.init_opt(mdp_spec=mdp_spec.high_mdp_spec, policy=policy.high_policy,
                                        baseline=baseline.high_baseline)
        with logger.tabular_prefix('Lo_'), logger.prefix('Lo | '):
            low_opt_info = \
                self.low_algo.init_opt(mdp_spec=mdp_spec.low_mdp_spec, policy=policy.low_policy,
                                       baseline=baseline.low_baseline)
        return dict(
            high=high_opt_info,
            low=low_opt_info
        )

    def optimize_policy(self, itr, policy, samples_data, opt_info, **kwargs):
        with logger.tabular_prefix('Hi_'), logger.prefix('Hi | '):
                opt_info["high"] = self.high_algo.optimize_policy(
                    itr, policy.high_policy, samples_data["high"], opt_info["high"])
        with logger.tabular_prefix('Lo_'), logger.prefix('Lo | '):
                opt_info["low"] = self.low_algo.optimize_policy(
                    itr, policy.low_policy, samples_data["low"], opt_info["low"])
        return opt_info

    @overrides
    def get_itr_snapshot(self, itr, mdp, policy, baseline, samples_data, opt_info, **kwargs):
        return dict(
            itr=itr,
            policy=policy,
            baseline=baseline,
            mdp=mdp,
        )

    def process_samples(self, itr, paths, mdp_spec, policy, baseline, bonus_evaluator, **kwargs):
        """
        Transform paths to high_paths and low_paths, and dispatch them to the high-level and low-level algorithms
        respectively.

        Also compute some diagnostic information:
        - I(a,g) = E_{s,g}[D_KL(p(a|g,s) || p(a|s))
                 = E_{s} [sum_g p(g|s) D_KL(p(a|g,s) || p(a|s))]
        """
        high_paths = []
        low_paths = []
        bonus_returns = []
        # Process the raw trajectories into appropriate high-level and low-level trajectories
        for path in paths:
            pdists = path['pdists']
            observations = path['observations']
            rewards = path['rewards']
            actions = path['actions']
            flat_observations = np.reshape(observations, (observations.shape[0], -1))
            high_pdists, low_pdists, subgoals = policy.split_pdists(pdists)
            high_path = dict(
                rewards=rewards,
                pdists=high_pdists,
                actions=subgoals,
                observations=observations,
            )
            # Right now the implementation assumes that high_pdists are given as probabilities of a categorical
            # distribution.
            # Need to be careful when this does not hold.
            assert isinstance(policy.high_policy, CategoricalMLPPolicy)
            # fill in information needed by bonus_evaluator
            path['subgoals'] = subgoals
            path['high_pdists'] = high_pdists
            bonuses = bonus_evaluator.predict(path)
            # TODO normalize these two terms
            # path['bonuses'] = bonuses
            low_rewards = rewards + self._mi_coeff * bonuses
            bonus_returns.append(np.sum(bonuses))
            low_path = dict(
                rewards=low_rewards,
                pdists=low_pdists,
                actions=actions,
                observations=np.concatenate([flat_observations, subgoals], axis=1)
            )
            high_paths.append(high_path)
            low_paths.append(low_path)
        logger.record_tabular("AverageBonusReturn", np.mean(bonus_returns))
        with logger.tabular_prefix('Hi_'), logger.prefix('Hi | '):
            high_samples_data = self.high_algo.process_samples(
                itr, high_paths, mdp_spec.high_mdp_spec, policy.high_policy, baseline.high_baseline)
        with logger.tabular_prefix('Lo_'), logger.prefix('Lo | '):
            low_samples_data = self.low_algo.process_samples(
                itr, low_paths, mdp_spec.low_mdp_spec, policy.low_policy, baseline.low_baseline)

        # Compute the mutual information I(a,g)
        # This is the component I'm still uncertain about how to abstract away yet
        high_observations = high_samples_data["observations"]
        # p(g|s)
        goal_probs = np.exp(policy.high_policy.get_pdists(high_observations))
        # p(a|g,s)
        action_given_goal_pdists = []
        # p(a|s) = sum_g p(g|s) p(a|g,s)
        action_pdists = 0
        # Actually compute p(a|g,s) and p(a|s)
        N = high_observations.shape[0]
        for goal in range(mdp_spec.n_subgoals):
            goal_onehot = np.tile(
                to_onehot(goal, mdp_spec.n_subgoals).reshape((1, -1)),
                (N, 1)
            )
            low_observations = np.concatenate([high_observations, goal_onehot], axis=1)
            action_given_goal_pdist = policy.low_policy.get_pdists(low_observations)
            action_given_goal_pdists.append(action_given_goal_pdist)
            action_pdists += goal_probs[:, [goal]] * action_given_goal_pdist
        # The mutual information between actions and goals
        mi_action_goal = 0
        for goal in range(mdp_spec.n_subgoals):
            mi_action_goal += np.mean(goal_probs[:, goal] * categorical_dist.kl(
                action_given_goal_pdists[goal],
                action_pdists
            ))
        # Log the mutual information
        logger.record_tabular("I(action,goal|state)", mi_action_goal)
        # We need to train the predictor for p(s'|g, s)
        with logger.prefix("MI | "), logger.tabular_prefix("MI_"):
            bonus_evaluator.fit(paths)
        return dict(
            high=high_samples_data,
            low=low_samples_data
        )
