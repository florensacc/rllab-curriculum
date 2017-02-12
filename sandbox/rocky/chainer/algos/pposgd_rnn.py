from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.misc import logger
from rllab.misc.overrides import overrides
from sandbox.rocky.chainer.algos.batch_polopt import BatchPolopt
from rllab.core.serializable import Serializable
import chainer.functions as F


class PPOSGD_rnn(BatchPolopt, Serializable):
    """
    PPOSGD for recurrent actor-critic policies.
    """

    def __init__(
            self,
            env,
            policy,
            baseline=None,
            optimizer=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if baseline is None:
            baseline = ZeroBaseline(env_spec=env.spec)
        BatchPolopt.__init__(self, env=env, policy=policy, baseline=baseline, **kwargs)


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
