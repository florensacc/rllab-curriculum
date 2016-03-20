from rllab.core.serializable import Serializable

class MDPSpec(Serializable):
    """
    This is a temporary solution until a better abstraction exists. It provides the necessary information to bootstraps
    a policy / baseline etc.
    """

    def __init__(
            self,
            observation_shape,
            observation_dtype,
            action_dim,
            action_dtype):
        Serializable.quick_init(self, locals())
        self._observation_shape = observation_shape
        self._observation_dtype = observation_dtype
        self._action_dim = action_dim
        self._action_dtype = action_dtype

    @property
    def observation_shape(self):
        return self._observation_shape

    @property
    def observation_dtype(self):
        return self._observation_dtype

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def action_dtype(self):
        return self._action_dtype
