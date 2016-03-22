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
            subgoal_interval=1,
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

    def _subsample_path(self, path, interval):
        pdists = path['pdists']
        observations = path['observations']
        rewards = path['rewards']
        actions = path['actions']
        path_length = len(pdists)
        chunked_length = int(np.ceil(path_length * 1.0 / interval))
        padded_length = chunked_length * interval
        padded_rewards = np.append(rewards, np.zeros(padded_length - path_length))
        chunked_rewards = np.sum(
            np.reshape(padded_rewards, (chunked_length, interval)),
            axis=1
        )
        chunked_pdists = pdists[::interval]
        chunked_actions = actions[::interval]
        chunked_observations = observations[::interval]
        return dict(
            observations=chunked_observations,
            actions=chunked_actions,
            pdists=chunked_pdists,
            rewards=chunked_rewards,
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
            path_length = len(pdists)
            observations = path['observations']
            rewards = path['rewards']
            actions = path['actions']
            flat_observations = np.reshape(observations, (observations.shape[0], -1))
            high_pdists, low_pdists, subgoals = policy.split_pdists(pdists)
            high_path = dict(
                observations=observations,
                actions=subgoals,
                pdists=high_pdists,
                rewards=rewards,
            )
            chunked_high_path = self._subsample_path(high_path, policy.subgoal_interval)
            # The high-level trajectory should be splitted according to the subgoal_interval parameter

            # high_path = dict(
            #     rewards=chunked_rewards,
            #     pdists=chunked_high_pdists,
            #     actions=chunked_subgoals,
            #     observations=chunked_observations,
            # )
            # Right now the implementation assumes that high_pdists are given as probabilities of a categorical
            # distribution.
            # Need to be careful when this does not hold.
            assert isinstance(policy.high_policy, CategoricalMLPPolicy)
            # fill in information needed by bonus_evaluator
            path['subgoals'] = subgoals#chunked_subgoals#subgoals
            path['high_pdists'] = high_pdists#chunked_high_pdists#high_pdists
            chunked_bonuses = bonus_evaluator.predict(chunked_high_path)
            bonuses = np.tile(
                np.expand_dims(chunked_bonuses, axis=1),
                (1, policy.subgoal_interval)
            ).flatten()[:path_length]
            # TODO normalize these two terms
            # path['bonuses'] = bonuses
            low_rewards = rewards + self._mi_coeff * bonuses
            if np.max(np.abs(bonuses)) > 1e3:
                import ipdb; ipdb.set_trace()
            bonus_returns.append(np.sum(bonuses))
            low_path = dict(
                observations=np.concatenate([flat_observations, subgoals], axis=1),
                actions=actions,
                pdists=low_pdists,
                rewards=low_rewards,
            )
            high_paths.append(chunked_high_path)
            low_paths.append(low_path)
        logger.record_tabular("AverageBonusReturn", np.mean(bonus_returns))
        with logger.tabular_prefix('Hi_'), logger.prefix('Hi | '):
            high_samples_data = self.high_algo.process_samples(
                itr, high_paths, mdp_spec.high_mdp_spec, policy.high_policy, baseline.high_baseline)
        with logger.tabular_prefix('Lo_'), logger.prefix('Lo | '):
            low_samples_data = self.low_algo.process_samples(
                itr, low_paths, mdp_spec.low_mdp_spec, policy.low_policy, baseline.low_baseline)

        mi_action_goal = self._compute_mi_action_goal(mdp_spec, policy, high_samples_data, low_samples_data)
        logger.record_tabular("I(action,goal|state)", mi_action_goal)
        # We need to train the predictor for p(s'|g, s)
        with logger.prefix("MI | "), logger.tabular_prefix("MI_"):
            bonus_evaluator.fit(high_paths)

        return dict(
            high=high_samples_data,
            low=low_samples_data
        )

    @staticmethod
    def _compute_mi_action_goal(mdp_spec, policy, high_samples_data, low_samples_data):
        if policy.high_policy.dist_family is categorical_dist \
                and policy.low_policy.dist_family is categorical_dist \
                and not policy.high_policy.is_recurrent \
                and not policy.low_policy.is_recurrent:

            # Compute the mutual information I(a,g)
            # This is the component I'm still uncertain about how to abstract away yet
            all_flat_observations = low_samples_data["observations"][:, :mdp_spec.observation_dim]
            high_observations = high_samples_data["observations"]
            # p(g|s)
            # shape: (N/subgoal_interval) * #subgoal
            chunked_goal_probs = np.exp(policy.high_policy.get_pdists(high_observations))
            goal_probs = np.tile(
                np.expand_dims(chunked_goal_probs, axis=1),
                (1, policy.subgoal_interval, 1),
            ).reshape((-1, mdp_spec.n_subgoals))
            # p(a|g,s)
            action_given_goal_pdists = []
            # p(a|s) = sum_g p(g|s) p(a|g,s)
            action_pdists = 0
            # Actually compute p(a|g,s) and p(a|s)
            N = all_flat_observations.shape[0]
            for goal in range(mdp_spec.n_subgoals):
                goal_onehot = np.tile(
                    to_onehot(goal, mdp_spec.n_subgoals).reshape((1, -1)),
                    (N, 1)
                )
                low_observations = np.concatenate([all_flat_observations, goal_onehot], axis=1)
                action_given_goal_pdist = np.exp(policy.low_policy.get_pdists(low_observations))
                action_given_goal_pdists.append(action_given_goal_pdist)
                action_pdists += goal_probs[:, [goal]] * action_given_goal_pdist
            # The mutual information between actions and goals
            mi_action_goal = 0
            for goal in range(mdp_spec.n_subgoals):
                mi_action_goal += np.mean(goal_probs[:, goal] * categorical_dist.kl(
                    np.log(action_given_goal_pdists[goal]),
                    np.log(action_pdists)
                ))
            if mi_action_goal > np.log(mdp_spec.n_subgoals):
                import ipdb;
                ipdb.set_trace()
            # Log the mutual information
            return mi_action_goal
        else:
            raise NotImplementedError
