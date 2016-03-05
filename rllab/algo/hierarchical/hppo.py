from rllab.algo.batch_polopt import BatchPolopt
from rllab.algo.ppo import PPO
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class HPPO(BatchPolopt):

    def __init__(self, high_algo, low_algo, **kwargs):
        self._high_algo = high_algo
        self._low_algo = low_algo
        super(HPPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self, mdp, policy, baseline, **kwargs):
        high_opt_info = \
            self._high_algo.init_opt(mdp=policy.high_mdp, policy=policy.high_policy, baseline=baseline.high_baseline)
        low_opt_info = \
            self._low_algo.init_opt(mdp=policy.low_mdp, policy=policy.low_policy, baseline=baseline.low_baseline)
        return dict(
            high=high_opt_info,
            low=low_opt_info
        )

    def optimize_policy(self, itr, policy, samples_data, opt_info):
        import ipdb; ipdb.set_trace()
        opt_info["high"] = self._high_opt.optimize_policy(
            itr, policy.high_policy, samples_data["high"], opt_info["high"])
        opt_info["low"] = self._low_opt.optimize_policy(
            itr, policy.high_policy, samples_data["low"], opt_info["low"])
        return opt_info

    @overrides
    def get_itr_snapshot(self, itr, mdp, policy, baseline, samples_data,
                         opt_info):
        return dict(
            itr=itr,
            policy=policy,
            baseline=baseline,
            mdp=mdp,
        )
