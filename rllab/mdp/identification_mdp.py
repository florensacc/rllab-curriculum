import numpy as np
from rllab.mdp.base import MDP, ControlMDP, SymbolicMDP
from rllab.core.serializable import Serializable
from rllab.mdp.proxy_mdp import ProxyMDP
from rllab.misc import autoargs
from rllab.misc.overrides import overrides
from rllab.misc.resolve import load_class


class IdentificationControlMDP(ProxyMDP, ControlMDP, Serializable):

    def __init__(self, mdp_cls, mdp_args):
        Serializable.quick_init(self, locals())
        self.mdp_cls = mdp_cls
        self.mdp_args = mdp_args
        mdp = self.gen_mdp()
        super(IdentificationControlMDP, self).__init__(mdp)
        ControlMDP.__init__(self)

    def gen_mdp(self):
        return self.mdp_cls.new_from_args(self.mdp_args, _silent=True, template_args=dict(noise=True))

    @overrides
    def reset(self):
        self._mdp = self.gen_mdp()
        return super(IdentificationControlMDP, self).reset()

