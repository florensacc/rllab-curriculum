from rllab.algos.batch_polopt import BatchPolopt
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class NOP(BatchPolopt):
    """
    NOP (no optimization performed) policy search algorithm
    """

    @autoargs.inherit(BatchPolopt.__init__)
    def __init__(
            self,
            **kwargs):
        super(NOP, self).__init__(**kwargs)

    @overrides
    def init_opt(self, mdp, policy, baseline):
        return {}

    @overrides
    def optimize_policy(self, itr, policy, samples_data, opt_info):
        return opt_info

    @overrides
    def get_itr_snapshot(self, itr, mdp, policy, baseline, samples_data,
                         opt_info):
        return {}
