from rllab.algo.batch_polopt import BatchPolopt
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable


class BatchHRL(BatchPolopt, Serializable):

    def __init__(self, high_algo, low_algo, **kwargs):
        super(BatchHRL, self).__init__(**kwargs)
        Serializable.quick_init(self, locals())
        self._high_algo = high_algo
        self._low_algo = low_algo

    @property
    def high_algo(self):
        return self._high_algo

    @property
    def low_algo(self):
        return self._low_algo

    @overrides
    def init_opt(self, mdp, policy, baseline, **kwargs):
        high_opt_info = \
            self.high_algo.init_opt(mdp=mdp.high_mdp, policy=policy.high_policy, baseline=baseline.high_baseline)
        low_opt_info = \
            self.low_algo.init_opt(mdp=mdp.low_mdp, policy=policy.low_policy, baseline=baseline.low_baseline)
        return dict(
            high=high_opt_info,
            low=low_opt_info
        )

    def optimize_policy(self, itr, policy, samples_data, opt_info, **kwargs):
        opt_info["high"] = self.high_algo.optimize_policy(
            itr, policy.high_policy, samples_data["high"], opt_info["high"])
        opt_info["low"] = self.low_algo.optimize_policy(
            itr, policy.high_policy, samples_data["low"], opt_info["low"])
        return opt_info

    @overrides
    def get_itr_snapshot(self, itr, mdp, policy, baseline, samples_data, opt_info, **kwargs):
        return dict(
            itr=itr,
            policy=policy,
            baseline=baseline,
            mdp=mdp,
        )

    def obtain_samples(self, itr, mdp, policy, baseline):
        samples_data = super(BatchHRL, self).obtain_samples(itr, mdp, policy, baseline)
        import ipdb; ipdb.set_trace()
