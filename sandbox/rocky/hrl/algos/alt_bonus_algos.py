


import numpy as np
from rllab.misc import special
from rllab.misc import ext
from rllab.misc import logger
from rllab.algos.batch_polopt import BatchPolopt
from rllab.misc import tensor_utils
from rllab.misc.overrides import overrides
from rllab.algos.npo import NPO
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable
from rllab.sampler.stateful_pool import singleton_pool
from rllab.sampler.utils import rollout
import theano
import theano.tensor as TT
import pickle as pickle

floatX = theano.config.floatX


def _worker_populate_test_env(G, test_env):
    G.test_env = pickle.loads(test_env)


def _worker_terminate_test_env(G):
    if getattr(G, "test_env", None):
        G.test_env.terminate()
        G.test_env = None


def populate_test_env(test_env):
    if singleton_pool.n_parallel > 1:
        singleton_pool.run_each(_worker_populate_test_env, [(pickle.dumps(test_env),)] * singleton_pool.n_parallel)
    else:
        singleton_pool.G.test_env = test_env


def terminate_test_env():
    singleton_pool.run_each(_worker_terminate_test_env, [tuple()] * singleton_pool.n_parallel)


def _worker_set_policy_params(G, params):
    G.policy.set_param_values(params)


def _worker_collect_one_test_path(G, max_path_length):
    path = rollout(G.test_env, G.policy, max_path_length)
    return path, len(path["rewards"])


class AltBonusBatchPolopt(BatchPolopt, Serializable):
    def __init__(
            self,
            bonus_evaluator,
            bonus_baseline,
            fit_before_evaluate=True,
            test_env=None,
            n_test_samples=None,
            test_max_path_length=None,
            *args,
            **kwargs):
        """
        :type bonus_evaluator: BonusEvaluator
        """
        Serializable.quick_init(self, locals())
        self.bonus_evaluator = bonus_evaluator
        self.bonus_baseline = bonus_baseline
        self.fit_before_evaluate = fit_before_evaluate
        self.adv_mean = theano.shared(np.cast[floatX](0.), "adv_mean")
        self.adv_std = theano.shared(np.cast[floatX](1.), "adv_std")
        self.bonus_adv_mean = theano.shared(np.cast[floatX](0.), "bonus_adv_mean")
        self.bonus_adv_std = theano.shared(np.cast[floatX](1.), "bonus_adv_std")
        super(AltBonusBatchPolopt, self).__init__(*args, **kwargs)
        self.test_env = test_env
        if test_env is not None:
            assert n_test_samples is not None, "Must specify n_test_samples if test_env is specified!"
            if test_max_path_length is None:
                test_max_path_length = self.max_path_length
        self.test_max_path_length = test_max_path_length
        self.n_test_samples = n_test_samples

    def start_worker(self):
        super(AltBonusBatchPolopt, self).start_worker()
        if self.test_env is not None:
            populate_test_env(self.test_env)

    def shutdown_worker(self):
        super(AltBonusBatchPolopt, self).shutdown_worker()
        if self.test_env is not None:
            terminate_test_env()

    def log_diagnostics(self, paths):
        super(AltBonusBatchPolopt, self).log_diagnostics(paths)
        self.bonus_evaluator.log_diagnostics(paths)
        self.bonus_baseline.log_diagnostics(paths)
        if self.test_env is not None:
            singleton_pool.run_each(
                _worker_set_policy_params,
                [(self.policy.get_param_values(),)] * singleton_pool.n_parallel
            )
            logger.log("Sampling trajectories for test environment")
            test_paths = singleton_pool.run_collect(
                _worker_collect_one_test_path,
                threshold=self.n_test_samples,
                args=(self.test_max_path_length,),
                show_prog_bar=True
            )
            for path in test_paths:
                path["returns"] = special.discount_cumsum(path["rewards"], self.discount)
            test_average_discounted_return = np.mean([path["returns"][0] for path in test_paths])
            test_average_return = np.mean([sum(path["rewards"]) for path in test_paths])
            logger.record_tabular("TestAverageReturn", test_average_return)
            logger.record_tabular("TestAverageDiscountedReturn", test_average_discounted_return)

    def process_samples(self, itr, paths):

        assert self.center_adv
        assert not self.positive_adv

        baselines = []
        returns = []
        bonus_baselines = []
        bonus_returns = []

        if self.fit_before_evaluate:
            self.bonus_evaluator.fit(paths)

        for path in paths:
            bonuses = self.bonus_evaluator.predict(path)
            # path["raw_rewards"] = path["rewards"]
            path["bonuses"] = bonuses
            # path["rewards"] = path["rewards"] + bonuses
            path_baselines = np.append(self.baseline.predict(path), 0)
            path_bonus_baselines = np.append(self.bonus_baseline.predict(dict(path, rewards=bonuses)), 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            bonus_deltas = path["bonuses"] + \
                           self.discount * path_bonus_baselines[1:] - \
                           path_bonus_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path["bonus_advantages"] = special.discount_cumsum(
                bonus_deltas, self.discount * self.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.discount)
            path["bonus_returns"] = special.discount_cumsum(path["bonuses"], self.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])
            bonus_baselines.append(path_bonus_baselines[:-1])
            bonus_returns.append(path["bonus_returns"])

        if not self.fit_before_evaluate:
            self.bonus_evaluator.fit(paths)

        if not self.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            bonuses = tensor_utils.concat_tensor_list([path["bonuses"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            bonus_advantages = tensor_utils.concat_tensor_list([path["bonus_advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])
            average_discounted_bonus_return = \
                np.mean([path["bonus_returns"][0] for path in paths])

            average_bonus = np.mean([np.mean(path["bonuses"]) for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.mean(self.policy.distribution.entropy(agent_infos))

            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages) + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            bonus_adv_mean = np.mean(bonus_advantages)
            bonus_adv_std = np.std(bonus_advantages) + 1e-8
            bonus_advantages = (bonus_advantages - bonus_adv_mean) / bonus_adv_std

            self.adv_mean.set_value(np.cast[floatX](adv_mean))
            self.adv_std.set_value(np.cast[floatX](adv_std))
            self.bonus_adv_mean.set_value(np.cast[floatX](bonus_adv_mean))
            self.bonus_adv_std.set_value(np.cast[floatX](bonus_adv_std))

            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )

            bonus_ev = special.explained_variance_1d(
                np.concatenate(bonus_baselines),
                np.concatenate(bonus_returns)
            )

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                advantages=advantages,
                bonuses=bonuses,
                bonus_advantages=bonus_advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = np.array([tensor_utils.pad_tensor(ob, max_path_length) for ob in obs])

            raw_adv = np.concatenate([path["advantages"] for path in paths])
            adv_mean = np.mean(raw_adv)
            adv_std = np.std(raw_adv) + 1e-8
            adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]

            adv = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            bonus_raw_adv = np.concatenate([path["bonus_advantages"] for path in paths])
            bonus_adv_mean = np.mean(bonus_raw_adv)
            bonus_adv_std = np.std(bonus_raw_adv) + 1e-8
            bonus_adv = [(path["bonus_advantages"] - bonus_adv_mean) / bonus_adv_std for path in paths]

            self.adv_mean.set_value(np.cast[floatX](adv_mean))
            self.adv_std.set_value(np.cast[floatX](adv_std))
            self.bonus_adv_mean.set_value(np.cast[floatX](bonus_adv_mean))
            self.bonus_adv_std.set_value(np.cast[floatX](bonus_adv_std))

            actions = [path["actions"] for path in paths]
            actions = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in actions])

            rewards = [path["rewards"] for path in paths]
            rewards = np.array([tensor_utils.pad_tensor(r, max_path_length) for r in rewards])

            bonuses = [path["bonuses"] for path in paths]
            bonuses = np.array([tensor_utils.pad_tensor(r, max_path_length) for r in bonuses])

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = np.array([tensor_utils.pad_tensor(v, max_path_length) for v in valids])

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            average_discounted_bonus_return = \
                np.mean([path["bonus_returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            average_bonus = np.mean([np.mean(path["bonuses"]) for path in paths])

            ent = np.mean(self.policy.distribution.entropy(agent_infos))

            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )

            bonus_ev = special.explained_variance_1d(
                np.concatenate(bonus_baselines),
                np.concatenate(bonus_returns)
            )

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                bonuses=bonuses,
                bonus_advantages=bonus_adv,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        logger.log("fitting baseline...")
        self.baseline.fit(paths)
        self.bonus_baseline.fit(paths)
        logger.log("fitted")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('AverageDiscountedBonusReturn',
                              average_discounted_bonus_return)
        logger.record_tabular('AverageBonus', average_bonus)
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('BonusExplainedVariance', bonus_ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data


class AltBonusNPO(AltBonusBatchPolopt):
    """
    Alternating Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            bonus_optimizer=None,
            bonus_optimizer_args=None,
            step_size=0.01,
            bonus_step_size=0.005,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        if bonus_optimizer is None:
            if bonus_optimizer_args is None:
                bonus_optimizer_args = dict()
            bonus_optimizer = PenaltyLbfgsOptimizer(**bonus_optimizer_args)
        self.optimizer = optimizer
        self.bonus_optimizer = bonus_optimizer
        self.step_size = step_size
        self.bonus_step_size = bonus_step_size
        super(AltBonusNPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        bonus_advantage_var = ext.new_tensor(
            'bonus_advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]
        ref_dist_info_vars = {
            k: ext.new_tensor(
                'ref_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        ref_dist_info_vars_list = [ref_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        bonus_kl = dist.kl_sym(ref_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)

        sym_bonus = self.bonus_evaluator.bonus_sym(obs_var, action_var, state_info_vars)
        sym_bonus = sym_bonus / self.adv_std

        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            bonus_mean_kl = TT.sum(bonus_kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
            bonus_surr_loss = - TT.sum(lr * bonus_advantage_var * valid_var +
                                       theano.gradient.zero_grad(lr) * sym_bonus * valid_var) / TT.sum(valid_var)
        else:
            mean_kl = TT.mean(kl)
            bonus_mean_kl = TT.mean(bonus_kl)
            surr_loss = - TT.mean(lr * advantage_var)
            bonus_surr_loss = - TT.mean(lr * bonus_advantage_var + theano.gradient.zero_grad(lr) * sym_bonus)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list
        bonus_input_list = [
                               obs_var,
                               action_var,
                               bonus_advantage_var,
                           ] + state_info_vars_list + old_dist_info_vars_list + ref_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)
            bonus_input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        self.bonus_optimizer.update_opt(
            loss=bonus_surr_loss,
            target=self.policy,
            leq_constraint=(bonus_mean_kl, self.bonus_step_size),
            inputs=bonus_input_list,
            constraint_name="bonus_mean_kl"
        )
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        all_bonus_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "bonus_advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        loss_before = self.optimizer.loss(all_input_values)
        self.optimizer.optimize(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)

        state_info_dict = {k: info for k, info in zip(self.policy.state_info_keys, state_info_list)}

        ref_dist_infos = self.policy.dist_info(samples_data["observations"], state_info_dict)
        ref_dist_info_list = [ref_dist_infos[k] for k in self.policy.distribution.dist_info_keys]

        all_bonus_input_values += tuple(state_info_list) + tuple(dist_info_list) + tuple(ref_dist_info_list)
        if self.policy.recurrent:
            all_bonus_input_values += (samples_data["valids"],)

        bonus_loss_before = self.bonus_optimizer.loss(all_bonus_input_values)
        self.bonus_optimizer.optimize(all_bonus_input_values)
        bonus_loss_after = self.bonus_optimizer.loss(all_bonus_input_values)

        mean_kl = self.optimizer.constraint_val(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('BonusLossBefore', bonus_loss_before)
        logger.record_tabular('BonusLossAfter', bonus_loss_after)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        logger.record_tabular('BonusdLoss', bonus_loss_before - bonus_loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            bonus_evaluator=self.bonus_evaluator,
            bonus_baseline=self.bonus_baseline,
            env=self.env,
        )


class AltBonusTRPO(AltBonusNPO):
    def __init__(self,
                 optimizer=None,
                 optimizer_args=None,
                 bonus_optimizer=None,
                 bonus_optimizer_args=None,
                 *args, **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        if bonus_optimizer is None:
            if bonus_optimizer_args is None:
                bonus_optimizer_args = dict()
            bonus_optimizer = ConjugateGradientOptimizer(**bonus_optimizer_args)
        AltBonusNPO.__init__(self, optimizer=optimizer, bonus_optimizer=bonus_optimizer, *args, **kwargs)
