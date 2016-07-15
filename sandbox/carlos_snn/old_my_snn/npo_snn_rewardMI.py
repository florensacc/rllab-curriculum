from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
# latent regressor to log the MI with other variables
from sandbox.carlos_snn.regressors.latent_regressor import Latent_regressor

# imports from batch_polopt I might need as not I use here process_samples and others
import numpy as np
from rllab.algos.base import RLAlgorithm
from rllab.sampler import parallel_sampler
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.sampler.utils import rollout
import itertools


# try to also plot the MC of the policy in the ec2 instance
# from sandbox.carlos_snn.plotters.plt_results2D import plot_all_exp

class NPO_snn(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            hallucinator=None,
            latent_regressor=None,
            reward_coef=0,
            self_normalize=False,
            log_individual_latents=False,  # to log the progress of each individual latent
            logged_MI=[],  # a list of tuples specifying the (obs,actions) that are regressed to find the latents
            n_samples=0,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.log_individual_latents = log_individual_latents

        self.hallucinator = hallucinator
        self.latent_regressor = latent_regressor
        self.reward_coef = reward_coef
        self.self_normalize = self_normalize
        self.n_samples = n_samples
        super(NPO_snn, self).__init__(**kwargs)

        # see what are the MI that want to be logged (it has to be done after initializing the super to have self.env)
        self.logged_MI = logged_MI
        if self.logged_MI == 'all_individual':
            self.logged_MI = []
            for o in range(self.env.spec.observation_space.flat_dim):
                self.logged_MI.append(([o], []))
            for a in range(self.env.spec.action_space.flat_dim):
                self.logged_MI.append(([], [a]))
        self.other_regressors = []
        if self.latent_regressor:  # check that there is a latent_regressor. there isn't if latent_dim=0.
            for reg_dict in self.logged_MI:  # this is poorly done, should fuse better the 2 dicts
                regressor_args = self.latent_regressor.regressor_args
                regressor_args['name'] = 'latent_reg_obs{}_act{}'.format(reg_dict['obs_regressed'], reg_dict['act_regressed'])
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
                for key, value in reg_dict.iteritems():
                    extra_regressor_args[key] = value
                temp_lat_reg = Latent_regressor(**extra_regressor_args)
                self.other_regressors.append(temp_lat_reg)
            pass

    # @overrides
    def process_samples(self, itr, paths):
        # save real undiscounted reward before changing them

        for i, path in enumerate(paths):
            if np.isnan(path['observations']).any():
                print 'The RAW observation of path {} have a NaN: '.format(i), path['observations'][0]
            if np.isnan(path['actions']).any() or np.isnan(path['agent_infos']['mean']).any():
                print 'The RAW actions of path {} have a Nan: '.format(i), path['actions'][0]
                print 'the params of the nn are: ', self.policy.get_param_values()
            if np.isnan(path['rewards']).any():
                print 'The RAW rewards of path {} have a Nan: '.format(i), path['rewards'][0]
        undiscounted_returns = [sum(path["rewards"]) for path in paths]
        logger.record_tabular('TrueAverageReturn', np.mean(undiscounted_returns))

        # If using a latent regressor (and possibly adding MI to the reward):
        if self.latent_regressor:
            with logger.prefix(' Latent_regressor '):
                self.latent_regressor.fit(paths)

                for i, path in enumerate(paths):
                    # if np.isnan(path['observations']).any():
                    #     print '(after reg.fit) The observation of path {} have a NaN: '.format(i), path['observations'][
                    #         0]
                    # if np.isnan(path['actions']).any():
                    #     print '(after reg.fit) The actions of path {} have a NaN: '.format(i), path['actions'][0]
                    # if np.isnan(path['rewards']).any():
                    #     print '(after reg.fit) The rewards of path {} have a Nan: '.format(i), path['rewards'][0]

                    path['logli_latent_regressor'] = self.latent_regressor.predict_log_likelihood(
                        [path], [path['agent_infos']['latents']])[0]  # this is for paths usually..

                    # if np.isnan(path['logli_latent_regressor']).any():
                    #     print 'The logli_latent_reg of path {} have NaN: '.format(i), path['logli_latent_regressor'][0]

                    # print "(after reg.pred) The latent sampled in path {} was: {}, " \
                    #       "the mean/actual action was {}{}, the probability of that one is: {}".format(
                    #         i, path['agent_infos']['latents'][0], path['agent_infos']['mean'][0],
                    #         path['actions'][0], path['logli_latent_regressor'][0])

                    # print path
                    path['true_rewards'] = list(path['rewards'])
                    path['rewards'] += self.reward_coef * path[
                        'logli_latent_regressor']  # the logli of the latent is the variable
                    # of the mutual information

        real_samples = ext.extract_dict(
            self.sampler.process_samples(itr, paths),
            # I don't need to process the hallucinated samples: the R, A,.. same!
            "observations", "actions", "advantages", "env_infos", "agent_infos"
        )
        real_samples["importance_weights"] = np.ones_like(real_samples["advantages"])

        # now, hallucinate some more...
        if self.hallucinator is None:
            return real_samples
        else:
            hallucinated = self.hallucinator.hallucinate(real_samples)
            if len(hallucinated) == 0:
                return real_samples
            all_samples = [real_samples] + hallucinated
            if self.self_normalize:
                all_importance_weights = np.asarray([x["importance_weights"] for x in all_samples])
                # It is important to use the mean instead of the sum. Otherwise, the computation of the weighted KL
                # divergence will be incorrect
                all_importance_weights = all_importance_weights / (np.mean(all_importance_weights, axis=0) + 1e-8)
                for sample, weights in zip(all_samples, all_importance_weights):
                    sample["importance_weights"] = weights
            return tensor_utils.concat_tensor_dict_list(all_samples)

    @overrides
    def train(self):
        self.start_worker()
        self.init_opt()
        episode_rewards = []
        episode_lengths = []
        for itr in xrange(self.current_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                paths = self.sampler.obtain_samples(itr)
                samples_data = self.process_samples(itr, paths)
                self.log_diagnostics(paths)
                self.optimize_policy(itr, samples_data)
                logger.log("saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)
                self.current_itr = itr + 1
                params["algo"] = self
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("saved")
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    self.update_plot()
                    if self.pause_for_plot:
                        raw_input("Plotting evaluation run: Press Enter to "
                                  "continue...")
        # # if working locally: we can plot at the same time
        # data_dir = logger.get_snapshot_dir()
        # plot_all_exp(data_dir)
        self.shutdown_worker()

    @overrides
    def log_diagnostics(self, paths):
        BatchPolopt.log_diagnostics(self, paths)
        if self.latent_regressor:
            with logger.prefix(
                    ' Latent regressor logging | '):  # this is mostly useless as log_diagnostics is only tabular
                self.latent_regressor.log_diagnostics(paths)
        # log the MI with other obs and action
        for i, lat_reg in enumerate(self.other_regressors):
            with logger.prefix(' Extra latent regressor {} | '.format(i)):  # same as above
                lat_reg.fit(paths)
                lat_reg.log_diagnostics(paths)

        if self.log_individual_latents and not self.policy.resample: # this is only valid for finite discrete latents!!
            all_latent_avg_returns = []
            clustered_by_latents = {}  # this could be done within the distribution to be more general, but ugly
            for path in paths:
                lat = str(path['agent_infos']['latents'][0])
                if lat not in clustered_by_latents:
                    clustered_by_latents[lat] = [path]
                else:
                    clustered_by_latents[lat].append(path)
            for latent_code, paths in clustered_by_latents.iteritems():
                with logger.tabular_prefix('new' + latent_code), logger.prefix(latent_code):
                    undiscounted_rewards = [sum(path["true_rewards"]) for path in paths]
                    all_latent_avg_returns.append(np.mean(undiscounted_rewards))
                    logger.record_tabular('Avg_TrueReturn', np.mean(undiscounted_rewards))
                    logger.record_tabular('Std_TrueReturn', np.std(undiscounted_rewards))
                    logger.record_tabular('Max_TrueReturn', np.max(undiscounted_rewards))
            with logger.tabular_prefix('all_lat_'), logger.prefix('all_lat_'):
                logger.record_tabular('MaxAvgReturn', np.max(all_latent_avg_returns))
                logger.record_tabular('MinAvgReturn', np.min(all_latent_avg_returns))
                logger.record_tabular('StdAvgReturn', np.std(all_latent_avg_returns))

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

        ## this will have to change as now the pdist depends also on the particuar latents var h sampled!
        # dist_info_vars = self.policy.dist_info_sym(obs_var, action_var)  ##returns dict with mean and log_std_var for this obs_var (action useless here!)
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
            loss=surr_loss,
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
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
