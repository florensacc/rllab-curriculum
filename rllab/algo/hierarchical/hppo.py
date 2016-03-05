from rllab.algo.batch_polopt import BatchPolopt
from rllab.algo.ppo import PPO
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class HPPO(BatchPolopt):

    @autoargs.inherit(PPO.__init__)
    def __init__(
            self,
            **kwargs):
        self._high_ppo = PPO(
            **kwargs
        )
        self._low_ppo = PPO(
            **kwargs
        )
        super(HPPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self, mdp, policy, baseline, **kwargs):
        high_opt_info = \
            self._high_ppo.init_opt(policy.high_mdp, policy.high_policy, baseline.high_baseline)
        low_opt_info = \
            self._low_ppo.init_opt(policy.low_mdp, policy.low_policy, baseline.low_baseline)
        return dict(
            high=high_opt_info,
            low=low_opt_info
        )

    def optimize_policy(self, itr, policy, samples_data, opt_info):
        opt_info["high"] = self._high_ppo.optimize_policy(
            itr, policy.high_policy, high_samples_data, opt_info["high"])
        opt_info["low"] = self._low_ppo.optimize_policy(
            itr, policy.high_policy, low_samples_data, opt_info["low"])
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
