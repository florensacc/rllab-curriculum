

from rllab.core.serializable import Serializable


class JointHallucinator(Serializable):
    """
    A joint hallucinator combines a couple different hallucinators together.
    """

    def __init__(self, hallucinators):
        Serializable.quick_init(self, locals())
        self.hallucinators = hallucinators

    def hallucinate(self, samples_data):
        return [x for h in self.hallucinators for x in h.hallucinate(samples_data)]
