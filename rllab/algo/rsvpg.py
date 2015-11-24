from rllab.misc import logger, autoargs
from rllab.misc.overrides import overrides
from rllab.algo.vpg import VPG


class RSVPG(VPG):
    """
    Risk-Seeking Vanilla Policy Gradient.
    """

    @autoargs.inherit(VPG.__init__)
    @autoargs.arg('q_threshold', type=float, help='Threshold for best q% paths')
    def __init__(
            self,
            q_threshold=0.2,
            **kwargs):
        self.q_threshold = q_threshold
        super(VPG, self).__init__(**kwargs)

    @overrides
    def obtain_samples(self, itr, mdp, policy, vf):
        results = super(RSVPG, self).obtain_samples(itr, mdp, policy, vf)
        pass
