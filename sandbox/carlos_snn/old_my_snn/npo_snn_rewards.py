from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt, BatchSampler
from rllab.algos.npo import NPO
from rllab.sampler import parallel_sampler

import rllab.misc.logger as logger
import theano
import theano.tensor as TT
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
# latent regressor to log the MI with other variables
from sandbox.carlos_snn.regressors.latent_regressor import Latent_regressor
from sandbox.carlos_snn.hallucinators.base import BaseHallucinator as Hallucinator
from sandbox.carlos_snn.distributions.categorical import from_index, from_onehot

# for logging hierarchy
from sandbox.carlos_snn.envs.hierarchized_snn_env import hierarchize_snn

# imports from batch_polopt I might need as not I use here process_samples and others
import numpy as np
from rllab.algos.base import RLAlgorithm
from rllab.sampler import parallel_sampler
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import os.path as osp
from rllab.sampler.utils import rollout
from sandbox.carlos_snn.sampler.utils_snn import rollout_snn
from sandbox.carlos_snn.sampler.utils import rollout as rollout_noEnvReset
import collections
import gc


class BatchSampler_snn(BatchSampler):
    def __init__(self,
                 *args,  # this collects algo, passing it to BatchSampler in the super __init__
                 reward_coef_mi=0,
                 reward_coef_bonus=None,
                 bonus_evaluator=None,
                 latent_regressor=None,
                 logged_MI=None,  # a list of tuples specifying the (obs,actions) that are regressed to find the latents
                 hallucinator=None,
                 n_hallu=0,
                 virtual_reset=False,
                 switch_lat_every=0,
                 **kwargs
                 ):
        """
        :type algo: BatchPolopt
        """
        super(BatchSampler_snn, self).__init__(*args, **kwargs)  # this should be giving a self.algo
        self.bonus_evaluator = bonus_evaluator if bonus_evaluator else []
        self.reward_coef_bonus = reward_coef_bonus if reward_coef_bonus else [0] * len(self.bonus_evaluator)
        self.reward_coef_mi = reward_coef_mi
        self.latent_regressor = latent_regressor
        self.logged_MI = logged_MI  # a list of tuples specifying the (obs,actions) that are regressed to find the latents
        self.hallucinator = hallucinator
        self.n_hallu = n_hallu
        self.virtual_reset = virtual_reset
        self.switch_lat_every = switch_lat_every

        # see what are the MI that want to be logged (it has to be done after initializing the super to have self.env)
        self.logged_MI = logged_MI if logged_MI else []
        if self.logged_MI == 'all_individual':
            self.logged_MI = []
            for o in range(self.algo.env.spec.observation_space.flat_dim):
                self.logged_MI.append(([o], []))
            for a in range(self.algo.env.spec.action_space.flat_dim):
                self.logged_MI.append(([], [a]))
        self.other_regressors = []
        if isinstance(self.latent_regressor, Latent_regressor):  # check that there is a latent_regressor. there isn't if latent_dim=0.
            for reg_dict in self.logged_MI:  # this is poorly done, should fuse better the 2 dicts
                regressor_args = self.latent_regressor.regressor_args
                regressor_args['name'] = 'latent_reg_obs{}_act{}'.format(reg_dict['obs_regressed'],
                                                                         reg_dict['act_regressed'])
                extra_regressor_args = {
                    'env_spec': self.latent_regressor.env_spec,
                    'policy': self.latent_regressor.policy,
                    'recurrent': reg_dict['recurrent'],
                    'predict_all': self.latent_regressor.predict_all,
                    'obs_regressed': self.latent_regressor.obs_regressed,
                    'act_regressed': self.latent_regressor.act_regressed,
                    'use_only_sign': self.latent_regressor.use_only_sign,
                    # 'optimizer': self.latent_regressor.optimizer,
                    'regressor_args': self.latent_regressor.regressor_args,
                }
                for key, value in reg_dict.items():
                    extra_regressor_args[key] = value
                temp_lat_reg = Latent_regressor(**extra_regressor_args)
                self.other_regressors.append(temp_lat_reg)

    def _worker_collect_one_path(self, G, max_path_length, scope=None):
        G = parallel_sampler._get_scoped_G(G, scope)
        path = rollout_snn(G.env, G.policy, max_path_length, switch_lat_every=self.switch_lat_every)
        return path, len(path["rewards"])


    def sample_paths(
            self,
            policy_params,
            max_samples,
            max_path_length=np.inf,
            env_params=None,
            scope=None):
        """
        :param policy_params: parameters for the policy. This will be updated on each worker process
        :param max_samples: desired maximum number of samples to be collected. The actual number of collected samples
        might be greater since all trajectories will be rolled out either until termination or until max_path_length is
        reached
        :param max_path_length: horizon / maximum length of a single trajectory
        :return: a list of collected paths
        """
        parallel_sampler.singleton_pool.run_each(
            parallel_sampler._worker_set_policy_params,
            [(policy_params, scope)] * parallel_sampler.singleton_pool.n_parallel
        )
        if env_params is not None:
            parallel_sampler.singleton_pool.run_each(
                parallel_sampler._worker_set_env_params,
                [(env_params, scope)] * parallel_sampler.singleton_pool.n_parallel
            )
        return parallel_sampler.singleton_pool.run_collect(
            self._worker_collect_one_path,
            threshold=max_samples,
            args=(max_path_length, scope),
            show_prog_bar=True
        )

    def obtain_samples(self, itr):
        cur_params = self.algo.policy.get_param_values()
        paths = self.sample_paths(
            policy_params=cur_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated

    @overrides
    def process_samples(self, itr, paths):
        # count visitations or whatever the bonus wants to do. This should not modify the paths
        for b_eval in self.bonus_evaluator:
            logger.log("fitting bonus evaluator before processing...")
            b_eval.fit_before_process_samples(paths)
            logger.log("fitted")
        # save real undiscounted reward before changing them
        undiscounted_returns = [sum(path["rewards"]) for path in paths]
        logger.record_tabular('TrueAverageReturn', np.mean(undiscounted_returns))
        for path in paths:
            path['true_rewards'] = list(path['rewards'])

        # If using a latent regressor (and possibly adding MI to the reward):
        if isinstance(self.latent_regressor, Latent_regressor):
            with logger.prefix(' Latent_regressor '):
                self.latent_regressor.fit(paths)

                for i, path in enumerate(paths):
                    path['logli_latent_regressor'] = self.latent_regressor.predict_log_likelihood(
                        [path], [path['agent_infos']['latents']])[0]  # this is for paths usually..

                    path['rewards'] += self.reward_coef_mi * path[
                        'logli_latent_regressor']  # the logli of the latent is the variable of the mutual information

        # for the extra bonus
        for b, b_eval in enumerate(self.bonus_evaluator):
            for i, path in enumerate(paths):
                bonuses = b_eval.predict(path)
                path['rewards'] += self.reward_coef_bonus[b] * bonuses

        real_samples = ext.extract_dict(
            BatchSampler.process_samples(self, itr, paths),
            # I don't need to process the hallucinated samples: the R, A,.. same!
            "observations", "actions", "advantages", "env_infos", "agent_infos"
        )
        real_samples["importance_weights"] = np.ones_like(real_samples["advantages"])

        # now, hallucinate some more...
        if isinstance(self.hallucinator, Hallucinator):
            hallucinated = self.hallucinator.hallucinate(real_samples)
            if len(hallucinated) == 0:
                return real_samples
            all_samples = [real_samples] + hallucinated
            if self.algo.self.algo.normalize:
                all_importance_weights = np.asarray([x["importance_weights"] for x in all_samples])
                # It is important to use the mean instead of the sum. Otherwise, the computation of the weighted KL
                # divergence will be incorrect
                all_importance_weights = all_importance_weights / (np.mean(all_importance_weights, axis=0) + 1e-8)
                for sample, weights in zip(all_samples, all_importance_weights):
                    sample["importance_weights"] = weights
            return tensor_utils.concat_tensor_dict_list(all_samples)
        else:
            return real_samples

    def log_diagnostics(self, paths):
        for b_eval in self.bonus_evaluator:
            b_eval.log_diagnostics(paths)

        if isinstance(self.latent_regressor, Latent_regressor):
            with logger.prefix(
                    ' Latent regressor logging | '):  # this is mostly useless as log_diagnostics is only tabular
                self.latent_regressor.log_diagnostics(paths)
        # log the MI with other obs and action
        for i, lat_reg in enumerate(self.other_regressors):
            with logger.prefix(' Extra latent regressor {} | '.format(i)):  # same as above
                lat_reg.fit(paths)
                lat_reg.log_diagnostics(paths)

class NPO_snn(NPO):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            # # this is for NPO! In fact we should inherit from NPO to avoid having this here
            # optimizer=None,
            # optimizer_args=None,
            # step_size=0.01,
            # some extra logging. What of this could be included in the sampler?
            log_individual_latents=False,  # to log the progress of each individual latent
            log_deterministic=False,  # log the performance of the policy with std=0 (for each latent separate)
            log_hierarchy=False,
            # possible modifications of the objective function with KL or L2 divergences between latents!
            L2_ub=1e6,
            KL_ub=1e6,
            reward_coef_l2=0,
            reward_coef_kl=0,
            # kwargs to the sampler (that also processes)
            reward_coef_mi=0,
            reward_coef_bonus=None,
            bonus_evaluator=None,
            latent_regressor=None,
            logged_MI=None,  # a list of tuples specifying the (obs,actions) that are regressed to find the latents
            hallucinator=None,
            n_hallu=0,
            virtual_reset=False,
            switch_lat_every=0,
            **kwargs):
        # # this should be just inherited from NPO
        # if optimizer is None:
        #     if optimizer_args is None:
        #         optimizer_args = dict()
        #     optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        # self.optimizer = optimizer
        # self.step_size = step_size
        # some logging
        self.log_individual_latents = log_individual_latents
        self.log_deterministic = log_deterministic
        self.log_hierarchy = log_hierarchy
        # bonus that are differentiable through
        self.reward_coef_l2 = reward_coef_l2
        self.L2_ub = L2_ub
        self.reward_coef_kl = reward_coef_kl
        self.KL_ub = KL_ub
        print(switch_lat_every)

        sampler_cls = BatchSampler_snn
        sampler_args = {'switch_lat_every': switch_lat_every,
                        'virtual_reset': virtual_reset,
                        'hallucinator': hallucinator,
                        'n_hallu': n_hallu,
                        'latent_regressor': latent_regressor,
                        'logged_MI': logged_MI,
                        'bonus_evaluator': bonus_evaluator,
                        'reward_coef_bonus': reward_coef_bonus,
                        'reward_coef_mi': reward_coef_mi,
                        }
        super(NPO_snn, self).__init__(sampler_cls=sampler_cls, sampler_args=sampler_args, **kwargs)

    @overrides
    def init_opt(self):
        assert not self.policy.recurrent
        is_recurrent = int(self.policy.recurrent)

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        importance_weights = TT.vector('importance_weights')  # for weighting the hallucinations
        ##
        latent_var = self.policy.latent_space.new_tensor_variable(
            'latents',
            extra_dims=1 + is_recurrent,
        )
        ##
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = self.policy.distribution  ### this can still be the dist P(a|s,__h__)
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,  ##define tensors old_mean and old_log_std
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]  ##put 2 tensors above in a list

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        ##CF
        dist_info_vars = self.policy.dist_info_sym(obs_var, latent_var)

        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        # if is_recurrent:
        #     mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
        #     surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        # else:
        mean_kl = TT.mean(kl * importance_weights)
        surr_loss = - TT.mean(lr * advantage_var * importance_weights)

        # # now we compute the kl with respect to all other possible latents:
        # list_all_latent_vars = []
        # for lat in range(self.policy.latent_dim):
        #     lat_shared_var = theano.shared(np.expand_dims(special.to_onehot(lat, self.policy.latent_dim), axis=0),
        #                                    name='latent_' + str(lat))
        #     list_all_latent_vars.append(lat_shared_var)
        #
        # list_dist_info_vars = []
        # all_l2 = []
        # list_l2 = []
        # all_kls = []
        # list_kls = []
        # for lat_var in list_all_latent_vars:
        #     expanded_lat_var = TT.tile(lat_var, [obs_var.shape[0], 1])
        #     list_dist_info_vars.append(self.policy.dist_info_sym(obs_var, expanded_lat_var))
        #
        # for dist_info_vars1 in list_dist_info_vars:
        #     list_l2_var1 = []
        #     list_kls_var1 = []
        #     for dist_info_vars2 in list_dist_info_vars:  # I'm doing the L2 without the sqrt!!
        #         list_l2_var1.append(TT.mean(TT.sqrt(TT.sum((dist_info_vars1['mean'] - dist_info_vars2['mean']) ** 2, axis=-1))))
        #         list_kls_var1.append(dist.kl_sym(dist_info_vars1, dist_info_vars2))
        #     all_l2.append(TT.stack(list_l2_var1))  # this is all the kls --> debug where it blows up!
        #     list_l2.append(TT.mean(TT.clip(list_l2_var1, 0, self.L2_ub)))
        #     all_kls.append(TT.stack(list_kls_var1))  # this is all the kls --> debug where it blows up!
        #     list_kls.append(TT.mean(TT.clip(list_kls_var1, 0, self.KL_ub)))
        #
        # if all_l2:  # if there was any latent:
        #     all_l2_stack = TT.stack(all_l2, axis=0)
        #     mean_clip_intra_l2 = TT.mean(list_l2)
        #
        #     self._mean_clip_intra_l2 = ext.compile_function(
        #         inputs=[obs_var],  # If you want to clip with the loss, you need to add here all input_list!!!
        #         outputs=[mean_clip_intra_l2]
        #     )
        #
        #     self._all_l2 = ext.compile_function(
        #         inputs=[obs_var],
        #         outputs=[all_l2_stack],
        #     )
        #
        # if all_kls:
        #     all_kls_stack = TT.stack(all_kls, axis=0)
        #     mean_clip_intra_kl = TT.mean(list_kls)
        #
        #     self._mean_clip_intra_kl = ext.compile_function(
        #         inputs=[obs_var],  # If you want to clip with the loss, you need to add here all input_list!!!
        #         outputs=[mean_clip_intra_kl]
        #     )
        #
        #     self._all_kls = ext.compile_function(
        #         inputs=[obs_var],
        #         outputs=[all_kls_stack],
        #     )
        #
        # if self.reward_coef_kl and self.reward_coef_l2:
        #     loss = surr_loss - self.reward_coef_kl * mean_clip_intra_kl - self.reward_coef_l2 * mean_clip_intra_l2
        # elif self.reward_coef_kl:
        #     loss = surr_loss - self.reward_coef_kl * mean_clip_intra_kl
        # elif self.reward_coef_l2:
        #     loss = surr_loss - self.reward_coef_l2 * mean_clip_intra_l2

        loss = surr_loss

        input_list = [  ##these are sym var. the inputs in optimize_policy have to be in same order!
                         obs_var,
                         action_var,
                         advantage_var,
                         importance_weights,
                         ##CF
                         latent_var,
                     ] + old_dist_info_vars_list  ##provide old mean and var, for the new states as they were sampled from it!
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    @overrides
    def optimize_policy(self, itr,
                        samples_data):  ###make that samples_data comes with latents: see train in batch_polopt
        all_input_values = tuple(ext.extract(  ### it will be in agent_infos!!! under key "latents"
            samples_data,
            "observations", "actions", "advantages", "importance_weights"
        ))
        agent_infos = samples_data["agent_infos"]
        ##CF
        all_input_values += (agent_infos[
                                 "latents"],)  # latents has already been processed and is the concat of all latents, but keeps key "latents"
        info_list = [agent_infos[k] for k in
                     self.policy.distribution.dist_info_keys]  ##these are the mean and var used at rollout, corresponding to
        all_input_values += tuple(info_list)  # old_dist_info_vars_list as symbolic var
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)

        loss_before = self.optimizer.loss(all_input_values)
        # this should always be 0. If it's not there is a problem.
        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        logger.record_tabular('MeanKL_Before', mean_kl_before)

        with logger.prefix(' PolicyOptimize | '):
            self.optimizer.optimize(all_input_values)

        mean_kl = self.optimizer.constraint_val(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def log_diagnostics(self, paths):
        BatchPolopt.log_diagnostics(self, paths)
        self.sampler.log_diagnostics(paths)

        if self.policy.latent_dim:

            # # extra logging
            # mean_clip_intra_kl = np.mean([self._mean_clip_intra_kl(path['observations']) for path in paths])
            # logger.record_tabular('mean_clip_intra_kl', mean_clip_intra_kl)

            # all_kls = [self._all_kls(path['observations']) for path in paths]

            # mean_clip_intra_l2 = np.mean([self._mean_clip_intra_l2(path['observations']) for path in paths])
            # logger.record_tabular('mean_clip_intra_l2', mean_clip_intra_l2)

            # all_l2 = [self._all_l2(path['observations']) for path in paths]
            # print('table of mean l2:\n', np.mean(all_l2, axis=0))

            if self.log_individual_latents and not self.policy.resample:  # this is only valid for finite discrete latents!!
                all_latent_avg_returns = []
                clustered_by_latents = collections.OrderedDict()  # this could be done within the distribution to be more general, but ugly
                for lat_key in range(self.policy.latent_dim):
                    # lat = from_index(i, self.policy.latent_dim)
                    clustered_by_latents[lat_key] = []
                for path in paths:
                    lat = path['agent_infos']['latents'][0]
                    lat_key = int(from_onehot(lat))  # from_onehot returns an axis less than the input.
                    clustered_by_latents[lat_key].append(path)

                for latent_key, paths in clustered_by_latents.items():  # what to do if this is empty?? set a default!
                    with logger.tabular_prefix(str(latent_key)), logger.prefix(str(latent_key)):
                        if paths:
                            undiscounted_rewards = [sum(path["true_rewards"]) for path in paths]
                        else:
                            undiscounted_rewards = [0]
                        all_latent_avg_returns.append(np.mean(undiscounted_rewards))
                        logger.record_tabular('Avg_TrueReturn', np.mean(undiscounted_rewards))
                        logger.record_tabular('Std_TrueReturn', np.std(undiscounted_rewards))
                        logger.record_tabular('Max_TrueReturn', np.max(undiscounted_rewards))
                        if self.log_deterministic:
                            lat = from_index(latent_key, self.policy.latent_dim)
                            with self.policy.fix_latent(lat), self.policy.set_std_to_0():
                                path_det = rollout(self.env, self.policy, self.max_path_length)
                                logger.record_tabular('Deterministic_TrueReturn', np.sum(path_det["rewards"]))

                with logger.tabular_prefix('all_lat_'), logger.prefix('all_lat_'):
                    logger.record_tabular('MaxAvgReturn', np.max(all_latent_avg_returns))
                    logger.record_tabular('MinAvgReturn', np.min(all_latent_avg_returns))
                    logger.record_tabular('StdAvgReturn', np.std(all_latent_avg_returns))

                if self.log_hierarchy:
                    max_in_path_length = 10
                    completed_in_paths = 0
                    path = rollout(self.env, self.policy, max_path_length=max_in_path_length, animated=False)
                    if len(path['rewards']) == max_in_path_length:
                        completed_in_paths += 1
                        for t in range(1, 50):
                            path = rollout_noEnvReset(self.env, self.policy, max_path_length=10, animated=False)
                            if len(path['rewards']) < 10:
                                break
                            completed_in_paths += 1
                    logger.record_tabular('Hierarchy', completed_in_paths)

        else:
            if self.log_deterministic:
                with self.policy.set_std_to_0():
                    path = rollout(self.env, self.policy, self.max_path_length)
                logger.record_tabular('Deterministic_TrueReturn', np.sum(path["rewards"]))
