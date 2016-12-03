from rllab.misc import logger
from rllab.misc.overrides import overrides
from sandbox.rocky.chainer.algos.batch_polopt import BatchPolopt
from rllab.core.serializable import Serializable
import chainer.functions as F

from sandbox.rocky.chainer.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class TRPO(BatchPolopt, Serializable):
    """
    Trust region policy optimization.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            step_size=0.01,
            entropy_bonus_coeff=0.,
            optimizer=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        BatchPolopt.__init__(self, env=env, policy=policy, baseline=baseline, **kwargs)
        self.step_size = step_size
        self.entropy_bonus_coeff = entropy_bonus_coeff
        if optimizer is None:
            optimizer = ConjugateGradientOptimizer()
        self.optimizer = optimizer
        self.optimizer.update_opt(
            target=policy,
            f_loss_constraint=self.loss_constraint_sym,
            max_constraint_val=step_size,
        )

    def loss_constraint_sym(self, *inputs):
        inputs = list(inputs)
        obs = inputs.pop(0)
        actions = inputs.pop(0)
        adv = inputs.pop(0)
        state_info_list = [inputs.pop(0) for _ in self.policy.state_info_keys]
        dist = self.policy.distribution
        old_dist_info_list = [inputs.pop(0) for _ in dist.dist_info_keys]
        state_infos = dict(zip(self.policy.state_info_keys, state_info_list))
        old_dist_infos = dict(zip(dist.dist_info_keys, old_dist_info_list))
        dist_info_vars = self.policy.dist_info_sym(obs, state_infos)
        N = len(obs)
        kl = dist.kl_sym(old_dist_infos, dist_info_vars)
        ent = dist.entropy_sym(dist_info_vars)
        lr = dist.likelihood_ratio_sym(actions, old_dist_infos, dist_info_vars)
        mean_kl = F.sum(kl) / N
        surr_loss = - (F.sum(lr * adv) - self.entropy_bonus_coeff * F.sum(ent)) / N
        return surr_loss, mean_kl

    @overrides
    def optimize_policy(self, itr, samples_data):
        obs = samples_data["observations"]
        actions = samples_data["actions"]
        adv = samples_data["advantages"]
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        old_dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]

        all_inputs = [obs, actions, adv] + state_info_list + old_dist_info_list

        logger.log("Computing loss & KL before")
        loss_before, mean_kl_before = self.optimizer.loss_constraint(all_inputs)
        logger.log("Optimizing")
        self.optimizer.optimize(all_inputs)
        logger.log("Computing loss & KL after")
        loss_after, mean_kl = self.optimizer.loss_constraint(all_inputs)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
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
