import numpy as np

from rllab.algos.batch_polopt import BatchPolopt
from rllab.core.serializable import Serializable
from rllab.distributions.categorical import Categorical
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.misc.special import to_onehot
from sandbox.rocky.hrl.misc import hrl_utils


class BatchHRL(BatchPolopt, Serializable):
    def __init__(
            self,
            env,
            policy,
            baseline,
            bonus_evaluator,
            high_algo,
            low_algo,
            mi_coeff=0.1,
            reward_coeff=1.,
            bonus_gradient=False,
            **kwargs):
        """
        :param env:
        :param policy:
        :param baseline:
        :param bonus_evaluator: Evaluator for the bonus
        :param high_algo: algorithm for high-level policy
        :param low_algo: algorithm for low-level policy
        :param mi_coeff: coefficient for mutual information
        :param reward_coeff: coefficient for the true reward (when training the low-level policy)
        :param bonus_gradient: whether to differentiate through the bonus reward
        :param kwargs:
        :return:
        """
        super(BatchHRL, self).__init__(env=env, policy=policy, baseline=baseline, **kwargs)
        Serializable.quick_init(self, locals())
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.bonus_evaluator = bonus_evaluator
        self.high_algo = high_algo
        self.low_algo = low_algo
        self.mi_coeff = mi_coeff
        self.reward_coeff = reward_coeff
        self.bonus_gradient = bonus_gradient

    @overrides
    def init_opt(self):
        with logger.tabular_prefix('Hi_'), logger.prefix('Hi | '):
            self.high_algo.init_opt()
        with logger.tabular_prefix('Lo_'), logger.prefix('Lo | '):
            self.low_algo.init_opt()

    def optimize_policy(self, itr, samples_data):
        with logger.tabular_prefix('Hi_'), logger.prefix('Hi | '):
            self.high_algo.optimize_policy(itr, samples_data["high"])
        with logger.tabular_prefix('Lo_'), logger.prefix('Lo | '):
            self.low_algo.optimize_policy(itr, samples_data["low"])

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )

    def log_diagnostics(self, paths):
        super(BatchHRL, self).log_diagnostics(paths)
        self.bonus_evaluator.log_diagnostics(paths)

    def process_samples(self, itr, paths):
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
        all_bonuses = []

        # self.env.analyzer.prepare_sym(paths, self.bonus_evaluator.component_idx)
        # self.bonus_evaluator.computed = False
        # self.bonus_evaluator.update_cache()

        with logger.prefix("MI | "), logger.tabular_prefix("MI_"):
            self.bonus_evaluator.fit(paths)

        for path in paths:
            rewards = path['rewards']
            actions = path['actions']
            high_agent_infos = path["agent_infos"]["high"]
            high_observations = path["agent_infos"]["high_obs"]
            low_agent_infos = path["agent_infos"]["low"]
            low_observations = path["agent_infos"]["low_obs"]
            subgoals = path["agent_infos"]["subgoal"]
            high_path = dict(
                observations=high_observations,
                actions=subgoals,
                env_infos=path["env_infos"],
                agent_infos=high_agent_infos,
                rewards=rewards,
            )
            chunked_high_path = hrl_utils.downsample_path(high_path, self.policy.subgoal_interval)
            bonuses = self.bonus_evaluator.predict(path)
            low_rewards = self.reward_coeff * rewards + self.mi_coeff * bonuses
            all_bonuses.extend(bonuses)
            bonus_returns.append(np.sum(bonuses))
            low_path = dict(
                observations=low_observations,
                actions=actions,
                env_infos=path["env_infos"],
                agent_infos=low_agent_infos,
                rewards=low_rewards,
            )
            high_paths.append(chunked_high_path)
            low_paths.append(low_path)
        logger.record_tabular("AverageBonusReturn", np.mean(bonus_returns))
        with logger.tabular_prefix('Hi_'), logger.prefix('Hi | '):
            high_samples_data = self.high_algo.process_samples(itr, high_paths)
        with logger.tabular_prefix('Lo_'), logger.prefix('Lo | '):
            low_samples_data = self.low_algo.process_samples(itr, low_paths)
            # for path in low_samples_data['paths']:
            #     if np.max(abs(self.baseline.low_baseline.predict(path))) > 1e3:
            #         import ipdb;
            #         ipdb.set_trace()

        # bonus_vals = self.bonus_evaluator.mi_bonus_sym().eval()
        # # compute bonus values explicitly to cross-compare
        # ref_bonus_vals = np.asarray(all_bonuses)
        #
        # diff = bonus_vals - ref_bonus_vals
        #
        # if np.linalg.norm(diff) > 1e-5:
        #     # compute bonus by hand to compare
        #     import ipdb; ipdb.set_trace()

        # mi_action_goal = self._compute_mi_action_goal(self.env.spec, self.policy, high_samples_data, low_samples_data)
        # logger.record_tabular("I(action,goal|state)", mi_action_goal)
        # We need to train the predictor for p(s'|g, s)



        return dict(
            high=high_samples_data,
            low=low_samples_data
        )

    @staticmethod
    def _compute_mi_action_goal(env_spec, policy, high_samples_data, low_samples_data):
        if isinstance(policy.high_policy.distribution, Categorical) \
                and isinstance(policy.low_policy.distribution, Categorical) \
                and not policy.high_policy.recurrent \
                and not policy.low_policy.recurrent:

            dist = Categorical()

            # Compute the mutual information I(a,g)
            # This is the component I'm still uncertain about how to abstract away yet
            all_flat_observations = low_samples_data["observations"][:, :(env_spec.observation_space.flat_dim +
                                                                          policy.subgoal_interval)]
            high_observations = high_samples_data["observations"]
            # p(g|s)
            # shape: (N/subgoal_interval) * #subgoal
            chunked_goal_probs = policy.high_policy.dist_info(high_observations, dict())["prob"]
            n_subgoals = policy.subgoal_space.n
            goal_probs = np.tile(
                np.expand_dims(chunked_goal_probs, axis=1),
                (1, policy.subgoal_interval, 1),
            ).reshape((-1, n_subgoals))
            # p(a|g,s)
            action_given_goal_pdists = []
            # p(a|s) = sum_g p(g|s) p(a|g,s)
            action_pdists = 0
            # Actually compute p(a|g,s) and p(a|s)
            N = all_flat_observations.shape[0]
            for goal in range(n_subgoals):
                goal_onehot = np.tile(
                    to_onehot(goal, n_subgoals).reshape((1, -1)),
                    (N, 1)
                )
                low_observations = np.concatenate([all_flat_observations, goal_onehot], axis=1)
                action_given_goal_pdist = policy.low_policy.dist_info(low_observations, dict())["prob"]
                action_given_goal_pdists.append(action_given_goal_pdist)
                action_pdists += goal_probs[:, [goal]] * action_given_goal_pdist
            # The mutual information between actions and goals
            mi_action_goal = 0
            for goal in range(n_subgoals):
                mi_action_goal += np.mean(goal_probs[:, goal] * dist.kl(
                    dict(prob=action_given_goal_pdists[goal]),
                    dict(prob=action_pdists)
                ))
            if mi_action_goal > np.log(n_subgoals):
                import ipdb;
                ipdb.set_trace()
            # Log the mutual information
            return mi_action_goal
        else:
            raise NotImplementedError
