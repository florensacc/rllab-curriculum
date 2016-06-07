from __future__ import print_function
from __future__ import absolute_import

from rllab.algos.trpo import TRPO
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import tensor_utils
from rllab.misc import special
from rllab.misc import logger
import numpy as np
import theano
import theano.tensor as TT


class TRPOBonus(TRPO, Serializable):
    def __init__(self, bonus_evaluator, mi_coeff=0.1, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.bonus_evaluator = bonus_evaluator
        self.mi_coeff = mi_coeff
        # These are used to rescale the symbolic reward bonus term
        self.adv_mean = theano.shared(np.float32(0.), "adv_mean")
        self.adv_std = theano.shared(np.float32(1.), "adv_std")
        super(TRPOBonus, self).__init__(*args, **kwargs)

    def init_opt(self):
        # super bad style, I know
        assert not self.policy.recurrent
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1,
            dtype=theano.config.floatX
        )
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        dist_info_vars = self.policy.dist_info_sym(obs_var, action_var)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)

        mean_kl = TT.mean(kl)

        sym_bonus = self.bonus_evaluator.mi_bonus_sym()
        sym_bonus = self.mi_coeff * sym_bonus / self.adv_std
        surr_loss = - TT.mean(lr * advantage_var) - TT.mean(theano.gradient.zero_grad(lr) * sym_bonus)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + old_dist_info_vars_list

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    def process_samples(self, itr, paths):

        baselines = []
        returns = []
        for path in paths:
            path_baselines = np.append(self.baseline.predict(path), 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        assert not self.positive_adv

        assert self.center_adv
        assert not self.policy.recurrent

        observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
        actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
        rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
        advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
        env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        self.adv_mean.set_value(np.float32(adv_mean))
        self.adv_std.set_value(np.float32(adv_std))

        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])

        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        ent = np.mean(self.policy.distribution.entropy(agent_infos))

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
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data
