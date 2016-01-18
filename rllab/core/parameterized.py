from rllab.core.serializable import Serializable
from rllab.misc.tensor_utils import unflatten_tensors


class Parameterized(object):

    @property
    def params(self):
        raise NotImplementedError

    @property
    def param_shapes(self):
        raise NotImplementedError

    @property
    def param_dtypes(self):
        raise NotImplementedError

    def get_param_values(self):
        raise NotImplementedError

    def set_param_values(self, flattened_params):
        raise NotImplementedError

    def flat_to_params(self, flattened_params):
        return unflatten_tensors(flattened_params, self.param_shapes)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.set_param_values(d["params"])
