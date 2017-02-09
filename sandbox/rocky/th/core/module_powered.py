from torch import nn

from sandbox.rocky.th.core.parameterized import Parameterized


class ModulePowered(Parameterized, nn.Module):
    def __init__(self):
        Parameterized.__init__(self)
        nn.Module.__init__(self)

    def get_params_internal(self, **tags):
        return list(self.parameters())
