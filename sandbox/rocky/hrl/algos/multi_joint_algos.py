from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from rllab.misc import special
from rllab.misc import logger
from rllab.misc import ext
from rllab.algos import util

from rllab.algos.base import RLAlgorithm
from rllab.sampler import parallel_sampler
from rllab.core.parameterized import Parameterized
# from rllab.algos.batch_polopt import BatchPolopt
# from rllab.algos.npo import NPO
from rllab.misc import tensor_utils
from rllab.misc.overrides import overrides
from rllab.algos.npo import NPO
from sandbox.rocky.hrl.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer

from rllab.core.serializable import Serializable
from sandbox.rocky.hrl.bonus_evaluators.base import BonusEvaluator
import theano
import theano.tensor as TT

floatX = theano.config.floatX


class JointParameterized(Parameterized):
    def __init__(self, components):
        super(JointParameterized, self).__init__()
        self.components = components

    def get_params_internal(self, **tags):
        params = [param for comp in self.components for param in comp.get_params_internal(**tags)]
        # only return unique parameters
        return sorted(set(params), key=hash)


class MultiJointBatchPolopt(RLAlgorithm, Serializable):
    def __init__(
            self,
            envs,
            policies,
            baselines,
            bonus_evaluators,
            scopes,
            reward_coeffs,
            batch_size=5000,
            max_path_length=500,
            n_itr=500,
            discount=0.99,
            gae_lambda=1,
            *args,
            **kwargs):
        """
        :type bonus_evaluators: list[BonusEvaluator]
        """
        Serializable.quick_init(self, locals())
        self.envs = envs
        self.policies = policies
        self.baselines = baselines
        self.scopes = scopes
        self.reward_coeffs = reward_coeffs
        self.bonus_evaluators = bonus_evaluators
        self.max_path_length = max_path_length
        self.batch_size = batch_size
        self.n_itr = n_itr

        assert len(envs) == len(policies) == len(baselines) == len(bonus_evaluators) == len(scopes) == len(
            reward_coeffs)
        self.adv_means = [theano.shared(np.cast[floatX](0.), "adv_mean_%d" % idx) for idx in range(len(envs))]
        self.adv_stds = [theano.shared(np.cast[floatX](1.), "adv_std_%d" % idx) for idx in range(len(envs))]
        self.discount = discount
        self.gae_lambda = gae_lambda
        super(MultiJointBatchPolopt, self).__init__(*args, **kwargs)

    def start_worker(self):
        for env, policy, scope in zip(self.envs, self.policies, self.scopes):
            with logger.tabular_prefix('%s_' % scope), logger.prefix('%s | ' % scope):
                parallel_sampler.populate_task(env, policy, scope=scope)

    def shutdown_worker(self):
        for scope in self.scopes:
            with logger.tabular_prefix('%s_' % scope), logger.prefix('%s | ' % scope):
                parallel_sampler.terminate_task(scope=scope)

    def init_opt(self):
        raise NotImplementedError

    def obtain_samples(self, itr):
        all_paths = dict()
        for policy, scope in zip(self.policies, self.scopes):
            with logger.tabular_prefix('%s_' % scope), logger.prefix('%s | ' % scope):
                cur_params = policy.get_param_values()
                paths = parallel_sampler.sample_paths(
                    policy_params=cur_params,
                    max_samples=self.batch_size,
                    max_path_length=self.max_path_length,
                    scope=scope,
                )
                all_paths[scope] = paths
        return all_paths

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        raise NotImplementedError

    def train(self):
        self.start_worker()
        self.init_opt()
        for itr in xrange(self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                paths = self.obtain_samples(itr)
                samples_data = self.process_samples(itr, paths)
                self.log_diagnostics(paths)
                self.optimize_policy(itr, samples_data)
                logger.log("saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)
                logger.save_itr_params(itr, params)
                logger.log("saved")
                logger.dump_tabular(with_prefix=False)

        self.shutdown_worker()

    def log_diagnostics(self, paths):
        for scope, env, policy, baseline, bonus_evaluator in zip(self.scopes, self.envs,
                                                                 self.policies, self.baselines, self.bonus_evaluators):
            with logger.tabular_prefix('%s_' % scope), logger.prefix('%s | ' % scope):
                env.log_diagnostics(paths[scope])
                policy.log_diagnostics(paths[scope])
                baseline.log_diagnostics(paths[scope])
                bonus_evaluator.log_diagnostics(paths[scope])

    def process_samples(self, itr, all_paths):

        all_samples_data = dict()

        for policy, baseline, bonus_evaluator, reward_coeff, adv_mean, adv_std, scope in zip(
                self.policies, self.baselines, self.bonus_evaluators, self.reward_coeffs, self.adv_means, self.adv_stds,
                self.scopes
        ):
            with logger.tabular_prefix('%s_' % scope), logger.prefix('%s | ' % scope):
                paths = all_paths[scope]

                baselines = []
                returns = []

                bonus_evaluator.fit(paths)

                for path in paths:
                    bonuses = bonus_evaluator.predict(path)
                    path["raw_rewards"] = path["rewards"]
                    path["bonuses"] = bonuses
                    path["rewards"] = reward_coeff * path["rewards"] + bonuses
                    path_baselines = np.append(baseline.predict(path), 0)
                    deltas = path["rewards"] + \
                             self.discount * path_baselines[1:] - \
                             path_baselines[:-1]
                    path["advantages"] = special.discount_cumsum(
                        deltas, self.discount * self.gae_lambda)
                    path["returns"] = special.discount_cumsum(path["rewards"], self.discount)
                    path["raw_returns"] = special.discount_cumsum(path["raw_rewards"], self.discount)
                    baselines.append(path_baselines[:-1])
                    returns.append(path["returns"])

                if not policy.recurrent:

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

                    ent = np.mean(policy.distribution.entropy(agent_infos))

                    adv_mean_val = np.mean(advantages)
                    adv_std_val = np.std(advantages) + 1e-8
                    advantages = (advantages - adv_mean_val) / adv_std_val

                    adv_mean.set_value(np.cast[floatX](adv_mean_val))
                    adv_std.set_value(np.cast[floatX](adv_std_val))

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
                else:
                    max_path_length = max([len(path["advantages"]) for path in paths])

                    # make all paths the same length (pad extra advantages with 0)
                    obs = [path["observations"] for path in paths]
                    obs = np.array([tensor_utils.pad_tensor(ob, max_path_length) for ob in obs])

                    # process advantages
                    raw_adv = np.concatenate([path["advantages"] for path in paths])
                    adv_mean_val = np.mean(raw_adv)
                    adv_std_val = np.std(raw_adv) + 1e-8
                    adv = [(path["advantages"] - adv_mean_val) / adv_std_val for path in paths]
                    adv = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

                    adv_mean.set_value(np.cast[floatX](adv_mean_val))
                    adv_std.set_value(np.cast[floatX](adv_std_val))

                    actions = [path["actions"] for path in paths]
                    actions = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in actions])

                    rewards = [path["rewards"] for path in paths]
                    rewards = np.array([tensor_utils.pad_tensor(r, max_path_length) for r in rewards])

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
                        np.mean([path["raw_returns"][0] for path in paths])

                    average_bonus = np.mean([np.mean(path["bonuses"]) for path in paths])

                    undiscounted_returns = [sum(path["raw_rewards"]) for path in paths]

                    ent = np.mean(policy.distribution.entropy(agent_infos))

                    ev = special.explained_variance_1d(
                        np.concatenate(baselines),
                        np.concatenate(returns)
                    )

                    samples_data = dict(
                        observations=obs,
                        actions=actions,
                        advantages=adv,
                        rewards=rewards,
                        valids=valids,
                        agent_infos=agent_infos,
                        env_infos=env_infos,
                        paths=paths,
                    )

                logger.log("fitting baseline...")
                baseline.fit(paths)
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

                all_samples_data[scope] = samples_data

        return all_samples_data


class MultiJointNPO(MultiJointBatchPolopt):
    def __init__(
            self,
            loss_weights,
            kl_weights,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            *args,
            **kwargs):
        MultiJointBatchPolopt.__init__(self, *args, **kwargs)
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        assert np.sum(kl_weights) > 1e-8
        assert np.sum(loss_weights) > 1e-8
        # normalize
        self.kl_weights = np.array(kl_weights) / np.sum(kl_weights)
        self.loss_weights = np.array(loss_weights) / np.sum(loss_weights)
        assert len(kl_weights) == len(loss_weights) == len(self.envs)

    @overrides
    def init_opt(self):

        all_input_list = []
        total_surr_loss = 0.
        total_mean_kl = 0.

        for env, policy, bonus_evaluator, adv_std, scope, loss_weight, kl_weight in zip(
                self.envs, self.policies, self.bonus_evaluators, self.adv_stds, self.scopes, self.loss_weights,
                self.kl_weights):
            is_recurrent = int(policy.recurrent)
            obs_var = env.observation_space.new_tensor_variable(
                '%s_obs' % scope,
                extra_dims=1 + is_recurrent,
            )
            action_var = env.action_space.new_tensor_variable(
                '%s_action' % scope,
                extra_dims=1 + is_recurrent,
            )
            advantage_var = ext.new_tensor(
                '%s_advantage' % scope,
                ndim=1 + is_recurrent,
                dtype=theano.config.floatX
            )
            dist = policy.distribution
            old_dist_info_vars = {
                k: ext.new_tensor(
                    '%s_old_%s' % (scope, k),
                    ndim=2 + is_recurrent,
                    dtype=theano.config.floatX
                ) for k in dist.dist_info_keys
                }
            old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

            state_info_vars = {
                k: ext.new_tensor(
                    '%s_%s' % (scope, k),
                    ndim=2 + is_recurrent,
                    dtype=theano.config.floatX
                ) for k in policy.state_info_keys
                }
            state_info_vars_list = [state_info_vars[k] for k in policy.state_info_keys]

            if is_recurrent:
                valid_var = TT.matrix('valid')
            else:
                valid_var = None

            dist_info_vars = policy.dist_info_sym(obs_var, state_info_vars)
            kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
            lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)

            sym_bonus = bonus_evaluator.bonus_sym(obs_var, action_var, state_info_vars)
            sym_bonus = sym_bonus / adv_std

            if is_recurrent:
                mean_kl = TT.sum(kl * valid_var) / (TT.sum(valid_var) + 1e-8)
                surr_loss = - TT.sum(lr * advantage_var * valid_var + theano.gradient.zero_grad(lr) * sym_bonus *
                                     valid_var) / (TT.sum(valid_var) + 1e-8)
            else:
                mean_kl = TT.mean(kl)
                surr_loss = - TT.mean(lr * advantage_var + theano.gradient.zero_grad(lr) * sym_bonus)

            total_surr_loss += loss_weight * surr_loss
            total_mean_kl += kl_weight * mean_kl

            input_list = [
                             obs_var,
                             action_var,
                             advantage_var,
                         ] + state_info_vars_list + old_dist_info_vars_list
            if is_recurrent:
                input_list.append(valid_var)

            all_input_list += input_list

        self.optimizer.update_opt(
            loss=total_surr_loss,
            target=JointParameterized(self.policies),
            leq_constraint=(total_mean_kl, self.step_size),
            inputs=all_input_list,
            constraint_name="mean_kl"
        )
        return dict()

    def get_itr_snapshot(self, itr, samples_data):
        params = dict(
            itr=itr,
            policy=self.policies[0],
            baseline=self.baselines[0],
            env=self.envs[0],
            bonus_evaluator=self.bonus_evaluators[0],
        )
        for policy, baseline, env, bonus_evaluator, scope in zip(self.policies, self.baselines, self.envs,
                                                                 self.bonus_evaluators, self.scopes):
            params[scope] = dict(
                policy=policy,
                baseline=baseline,
                env=env,
                bonus_evaluator=bonus_evaluator,
            )
        return params

    @overrides
    def optimize_policy(self, itr, samples_data):
        total_all_input_values = []
        grouped_inputs = []
        for policy, scope in zip(self.policies, self.scopes):
            all_input_values = tuple(ext.extract(
                samples_data[scope],
                "observations", "actions", "advantages"
            ))
            agent_infos = samples_data[scope]["agent_infos"]
            state_info_list = [agent_infos[k] for k in policy.state_info_keys]
            dist_info_list = [agent_infos[k] for k in policy.distribution.dist_info_keys]
            all_input_values += tuple(state_info_list) + tuple(dist_info_list)
            if policy.recurrent:
                all_input_values += (samples_data[scope]["valids"],)
            total_all_input_values += all_input_values
            grouped_inputs.append(all_input_values)
        loss_before = self.optimizer.loss(total_all_input_values)
        mean_kl_before = self.optimizer.constraint_val(total_all_input_values)
        if isinstance(self.optimizer, ConjugateGradientOptimizer):
            self.optimizer.optimize(total_all_input_values, subsample_grouped_inputs=grouped_inputs)
        else:
            self.optimizer.optimize(total_all_input_values)
        mean_kl = self.optimizer.constraint_val(total_all_input_values)
        loss_after = self.optimizer.loss(total_all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()


class MultiJointTRPO(MultiJointNPO):
    def __init__(self,
                 optimizer=None,
                 optimizer_args=None,
                 *args, **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        MultiJointNPO.__init__(self, optimizer=optimizer, *args, **kwargs)
