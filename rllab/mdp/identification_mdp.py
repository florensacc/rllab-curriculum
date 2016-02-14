from rllab.core.serializable import Serializable
from rllab.mdp.proxy_mdp import ProxyMDP
from rllab.misc.overrides import overrides


class IdentificationMDP(ProxyMDP, Serializable):

    def __init__(self, mdp_cls, mdp_args):
        Serializable.quick_init(self, locals())
        self.mdp_cls = mdp_cls
        self.mdp_args = mdp_args
        mdp = self.gen_mdp()
        super(IdentificationMDP, self).__init__(mdp)

    def gen_mdp(self):
        return self.mdp_cls.new_from_args(self.mdp_args, _silent=True, template_args=dict(noise=True))

    @overrides
    def reset(self):
        if getattr(self, "_mdp", None):
            if hasattr(self._mdp, "release"):
                self._mdp.release()
        self._mdp = self.gen_mdp()
        return super(IdentificationMDP, self).reset()
