from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
#imports from batch_polopt I might need as not I use here process_samples and others
import numpy as np
from rllab.algos.base import RLAlgorithm
from rllab.sampler import parallel_sampler
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger
import rllab.plotter as plotter


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

        self.hallucinator = hallucinator
        self.latent_regressor = latent_regressor
        self.reward_coef = reward_coef
        self.self_normalize = self_normalize
        self.n_samples=n_samples
        super(NPO_snn, self).__init__(**kwargs)

    @overrides
    def process_samples(self, itr, paths):
        # save real undiscounted reward before changing them
        undiscounted_returns = [sum(path["rewards"]) for path in paths]
        logger.record_tabular('TrueAverageReturn', np.mean(undiscounted_returns))
        #check the latents
        if self.latent_regressor:
            self.latent_regressor.fit(paths)
            # for path in paths:
            #     print 'the action: ', path['actions']
            #     print 'the latents: ', path['agent_infos']['latents']
            #     print 'the regressor distr: ', self.latent_regressor.get_output_p(path)
            #     print 'latents entropy: ', self.policy.latent_dist.entropy(self.policy.latent_dist_info_vars)
            #     print 'mutual info lb: ', self.latent_regressor.lowb_mutual(paths)

            for path in paths:
                path['logli_latent_regressor'] = self.latent_regressor.predict_log_likelihood(
                                                    [path], [path['agent_infos']['latents']])[0] #this is for paths usually..
                path['true_rewards'] = path['rewards']
                path['rewards'] += self.reward_coef * path['logli_latent_regressor']  # the logli of the latent is the variable
                                                                                        # of the mutual information

        real_samples = ext.extract_dict(
            BatchPolopt.process_samples(self, itr, paths),   # I don't need to process the hallucinated samples: the R, A,.. same!
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
        for itr in xrange(self.start_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                paths = self.obtain_samples(itr)
                samples_data = self.process_samples(itr, paths)
                self.log_diagnostics(paths)
                self.optimize_policy(itr, samples_data)
                logger.log("saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
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

        self.shutdown_worker()

    @overrides
    def log_diagnostics(self, paths):
        BatchPolopt.log_diagnostics(self, paths) # call the diagnost of env, policy and baseline
        if self.latent_regressor:
            self.latent_regressor.log_diagnostics(paths)

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
        dist = self.policy.distribution   ### this can still be the dist P(a|s,__h__)
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,   ##define tensors old_mean and old_log_std
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys] ##put 2 tensors above in a list

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

        input_list = [                  ##these are sym var. the inputs in optimize_policy have to be in same order!
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
    def optimize_policy(self, itr, samples_data):   ###make that samples_data comes with latents: see train in batch_polopt
        all_input_values = tuple(ext.extract(       ### it will be in agent_infos!!! under key "latents"
            samples_data,
            "observations",  "actions", "advantages", "importance_weights"
        ))
        agent_infos = samples_data["agent_infos"]
        ##CF
        all_input_values += (agent_infos["latents"],)  #latents has already been processed and is the concat of all latents, but keeps key "latents"
        #
        info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys] ##these are the mean and var used at rollout, corresponding to
        all_input_values += tuple(info_list)                                            # old_dist_info_vars_list as symbolic var
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)

        loss_before = self.optimizer.loss(all_input_values)
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