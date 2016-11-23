from sandbox.rocky.chainer.core.parameterized import Parameterized
import chainer


class LinkPowered(Parameterized, chainer.Link):
    def get_params_internal(self, **tags):
        return list(self.params())
