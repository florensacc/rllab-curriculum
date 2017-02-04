from sandbox.rocky.chainer.core.parameterized import Parameterized
import chainer


class LinkPowered(Parameterized, chainer.Link):
    def get_params_internal(self, **tags):
        return [x for _, x in sorted(list(self.namedparams()))]
