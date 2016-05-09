from __future__ import print_function
from __future__ import absolute_import

from rllab.algos.batch_polopt import BatchPolopt
from rllab.algos.npo import NPO
from rllab.algos.trpo import TRPO
from rllab.algos.ppo import PPO
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import logger
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.misc import tensor_utils
import theano
import numpy as np
import theano.tensor as TT


class BatchPolopt_snn_test(BatchPolopt, Serializable):
    def __init__(self, hallucinator=None, self_normalize=False, n_samples=0, *args, **kwargs):
        """
        :param hallucinator: Hallucinator object used for generating extra samples
        :param self_normalize: whether to normalize the importance weights to sum to one for the same real experience
        :return:
        """
        Serializable.quick_init(self, locals())
        self.hallucinator = hallucinator
        self.self_normalize = self_normalize
        self.n_samples = n_samples
        BatchPolopt.__init__(self, *args, **kwargs)


    def process_samples(self, itr, paths):
        n_original = len(paths)
        for i, path in enumerate(paths[:n_original]):
            for _ in range(self.n_samples):
                paths.append(path)
        if True:
            samples_data = BatchPolopt.process_samples(self,itr,paths)
            samples_data["importance_weights"] = np.ones_like(samples_data["advantages"])
            return samples_data
## ------------------------------
        real_samples = ext.extract_dict(
            BatchPolopt.process_samples(self, itr, paths),
            "observations", "actions", "advantages", "env_infos", "agent_infos"
        )
        real_samples["importance_weights"] = np.ones_like(real_samples["advantages"])
        if True:
            real_samples = [real_samples]*self.n_samples
            print (real_samples)
            return tensor_utils.concat_tensor_dict_list(real_samples)
## -------------------------------

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


class NPO_snn_test(NPO, BatchPolopt_snn_test):
    def __init__(self, *args, **kwargs):
        BatchPolopt_snn_test.__init__(self, *args, **kwargs)
        NPO.__init__(self, *args, **kwargs)

    def init_opt(self):
        assert not self.policy.recurrent
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        importance_weights = TT.vector('importance_weights')
        advantage_var = TT.vector('advantage')

        dist = self.policy.distribution
        old_dist_info_vars = {
            k: TT.matrix('old_%s' % k)
            for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: TT.matrix(k)
            for k in self.policy.state_info_keys
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        mean_kl = TT.mean(kl * importance_weights)
        surr_loss = - TT.mean(lr * advantage_var * importance_weights)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                         importance_weights,
                     ] + state_info_vars_list + old_dist_info_vars_list

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages", "importance_weights"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        loss_before = self.optimizer.loss(all_input_values)
        self.optimizer.optimize(all_input_values)
        mean_kl = self.optimizer.constraint_val(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
            hallucinator=self.hallucinator,
        )


class TRPO_snn_test(NPO_snn_test):
    def __init__(self,
                 optimizer=None,
                 optimizer_args=None,
                 *args, **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        NPO_snn_test.__init__(self, optimizer=optimizer, *args, **kwargs)
