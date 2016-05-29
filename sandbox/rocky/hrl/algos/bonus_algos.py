from __future__ import print_function
from __future__ import absolute_import
import numpy as np
from rllab.misc import special
from rllab.misc import logger
from rllab.misc import ext
from rllab.algos import util

from rllab.algos.batch_polopt import BatchPolopt
from rllab.misc import tensor_utils
from rllab.misc.overrides import overrides
from rllab.algos.npo import NPO
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

from rllab.core.serializable import Serializable
from sandbox.rocky.hrl.bonus_evaluators.base import BonusEvaluator
import theano
import theano.tensor as TT


floatX = theano.config.floatX


class BonusBatchPolopt(BatchPolopt, Serializable):
    def __init__(
            self,
            bonus_evaluator,
            fit_before_evaluate=True,
            *args,
            **kwargs):
        """
        :type bonus_evaluator: BonusEvaluator
        """
        Serializable.quick_init(self, locals())
        self.bonus_evaluator = bonus_evaluator
        self.fit_before_evaluate = fit_before_evaluate
        self.adv_mean = theano.shared(np.cast[floatX](0.), "adv_mean")
        self.adv_std = theano.shared(np.cast[floatX](1.), "adv_std")
        super(BonusBatchPolopt, self).__init__(*args, **kwargs)

    def log_diagnostics(self, paths):
        super(BonusBatchPolopt, self).log_diagnostics(paths)
        self.bonus_evaluator.log_diagnostics(paths)

    def process_samples(self, itr, paths):

        assert self.center_adv
        assert not self.positive_adv

        baselines = []
        returns = []

        if self.fit_before_evaluate:
            self.bonus_evaluator.fit(paths)

        for path in paths:
            bonuses = self.bonus_evaluator.predict(path)
            path["raw_rewards"] = path["rewards"]
            path["bonuses"] = bonuses
            path["rewards"] = path["rewards"] + bonuses
            path_baselines = np.append(self.baseline.predict(path), 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.discount)
            path["raw_returns"] = special.discount_cumsum(path["raw_rewards"], self.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        if not self.fit_before_evaluate:
            self.bonus_evaluator.fit(paths)

        observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
        actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
        rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
        advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
        env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

        average_discounted_return = \
            np.mean([path["raw_returns"][0] for path in paths])

        average_bonus = np.mean([np.mean(path["bonuses"]) for path in paths])

        undiscounted_returns = [sum(path["raw_rewards"]) for path in paths]

        ent = np.mean(self.policy.distribution.entropy(agent_infos))

        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = advantages - adv_mean

        self.adv_mean.set_value(np.cast[floatX](adv_mean))
        self.adv_std.set_value(np.cast[floatX](adv_std))

        ev = special.explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(returns)
        )

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
            paths=paths,
        )

        logger.log("fitting baseline...")
        self.baseline.fit(paths)
        logger.log("fitted")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('AverageBonus', average_bonus)
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data


class BonusNPO(NPO, BonusBatchPolopt):
    def __init__(self, *args, **kwargs):
        NPO.__init__(self, *args, **kwargs)
        BonusBatchPolopt.__init__(self, *args, **kwargs)

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
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

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
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)

        sym_bonus = self.bonus_evaluator.bonus_sym(obs_var, action_var, state_info_vars)
        sym_bonus = sym_bonus / self.adv_std

        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * advantage_var * valid_var + theano.gradient.zero_grad(lr) * sym_bonus *
                                 valid_var) / TT.sum(valid_var)
        else:
            mean_kl = TT.mean(kl)
            surr_loss = - TT.mean(lr * advantage_var + theano.gradient.zero_grad(lr) * sym_bonus)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
            bonus_evaluator=self.bonus_evaluator,
        )


class BonusTRPO(BonusNPO):
    def __init__(self,
                 optimizer=None,
                 optimizer_args=None,
                 *args, **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        BonusNPO.__init__(self, optimizer=optimizer, *args, **kwargs)
