from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.core.serializable import Serializable

class SubgoalBaseline(ZeroBaseline, Serializable):

    def __init__(self, env_spec, high_baseline, low_baseline):
        Serializable.quick_init(self, locals())
        super(SubgoalBaseline, self).__init__(env_spec)
        # TODO is there any way to avoid repeating this construction?
        self._high_baseline = high_baseline
        self._low_baseline = low_baseline

    @property
    def high_baseline(self):
        return self._high_baseline

    @property
    def low_baseline(self):
        return self._low_baseline
