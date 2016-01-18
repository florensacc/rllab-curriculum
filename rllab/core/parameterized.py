from rllab.core.serializable import Serializable


class Parameterized(object):

    @property
    def trainable_params(self):
        """
        Get the list of parameters that should be included in the optimization.
        """
        raise NotImplementedError

    def get_trainable_param_values(self):
        raise NotImplementedError

    def set_trainable_param_values(self, flattened_params):
        raise NotImplementedError

    @property
    def trainable_param_shapes(self):
        raise NotImplementedError

    @property
    def trainable_param_dtypes(self):
        raise NotImplementedError

    @property
    def params(self):
        """
        Get the list of all parameters. This is called when performing target
        network updates, or when serializing the parameterized object.
        """
        return self.trainable_params

    def get_param_values(self):
        return self.get_trainable_param_values()

    def set_param_values(self, flattened_params):
        self.set_trainable_param_values(flattened_params)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.set_param_values(d["params"])
