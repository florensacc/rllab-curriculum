from rllab.misc import logger
from rllab.misc.overrides import overrides
from sandbox.rocky.chainer.algos.batch_polopt import BatchPolopt
from rllab.core.serializable import Serializable
import chainer
import chainer.functions as F
import numpy as np


class VPG(BatchPolopt, Serializable):
    """
    Vanilla Policy Gradient.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            learning_rate=1e-3,
            **kwargs):
        Serializable.quick_init(self, locals())
        self.optimizer = chainer.optimizers.Adam(alpha=learning_rate)
        self.optimizer.setup(policy)
        super(VPG, self).__init__(env=env, policy=policy, baseline=baseline, **kwargs)

    def loss_sym(self, samples_data):
        obs = samples_data["observations"]
        actions = samples_data["actions"]
        adv = samples_data["advantages"]
        agent_infos = samples_data["agent_infos"]
        dist_infos = self.policy.dist_info_sym(obs, agent_infos)
        logli = self.policy.distribution.log_likelihood_sym(actions, dist_infos)
        loss = -F.sum(logli * adv) / len(logli.data)
        return loss

    def kl_sym(self, samples_data):
        agent_infos = samples_data["agent_infos"]
        obs = samples_data["observations"]
        dist_infos = self.policy.dist_info_sym(obs, agent_infos)
        return self.policy.distribution.kl_sym(agent_infos, dist_infos)

    @overrides
    def optimize_policy(self, itr, samples_data):
        self.optimizer.zero_grads()
        loss = self.loss_sym(samples_data)
        loss.backward()
        loss_before = loss.data
        self.optimizer.update()
        loss_after = self.loss_sym(samples_data).data
        kls = self.kl_sym(samples_data).data
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)
        logger.record_tabular('MeanKL', np.mean(kls))
        logger.record_tabular('MaxKL', np.max(kls))

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
