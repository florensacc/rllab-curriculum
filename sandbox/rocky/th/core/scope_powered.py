from sandbox.rocky.th import ops
from sandbox.rocky.th.core.parameterized import Parameterized


class ScopePowered(Parameterized):
    def __init__(self, scope):
        self._scope = scope
        Parameterized.__init__(self)

    def get_params_internal(self, **tags):
        trainable = tags.get('trainable', None)
        regularizable = tags.get('regularizable', None)
        params = ops.scoped_variables(self._scope)
        params = sorted(params, key=lambda x: x._ops_full_name)
        if trainable is True:
            params = [p for p in params if ops.is_trainable(p)]
        elif trainable is False:
            params = [p for p in params if not ops.is_trainable(p)]
        if regularizable is True:
            params = [p for p in params if ops.is_regularizable(p)]
        elif regularizable is False:
            params = [p for p in params if not ops.is_regularizable(p)]
        return params
