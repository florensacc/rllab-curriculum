from collections import OrderedDict
from contextlib import contextmanager

import torch

from sandbox.rocky.th.ops import conversion_ops

_funcs = OrderedDict()
_fn_type = type(lambda x: x)


def register(name=None):
    def _register(fn):
        if name is None or isinstance(name, _fn_type):
            fn_name = fn.__name__
        else:
            fn_name = name
        _funcs[fn_name] = fn
        return fn

    if isinstance(name, _fn_type):
        return _register(name)
    return _register


@contextmanager
def registering(name, fn):
    if name in _funcs:
        has_prev = True
        prev_fn = _funcs[name]
    else:
        has_prev = False
        prev_fn = None
    _funcs[name] = fn
    yield
    if has_prev:
        _funcs[name] = prev_fn
    else:
        del _funcs[name]


class Wrapper(object):
    def __init__(self, value_or_values, mode='concat', concat_dim=-1, parent=None):
        if isinstance(value_or_values, (tuple, list)):
            value_or_values = [conversion_ops.as_variable(
                x) for x in value_or_values]
            if mode == 'concat':
                if concat_dim < 0:
                    concat_dim += value_or_values[0].dim()
                value = torch.cat(value_or_values, concat_dim)
            elif mode == 'sum':
                value = sum(value_or_values)
            else:
                raise NotImplementedError
        else:
            value = conversion_ops.as_variable(value_or_values)
        self._value = value
        self._parent = parent

    @property
    def value(self):
        return self._value

    def __getattr__(self, item):
        if item in _funcs:
            return lambda *args, **kwargs: Wrapper(
                _funcs[item](self.value, *args, **kwargs),
                parent=self._parent
            )
        else:
            return super().__getattribute__(item)

    def branch_out(self):
        return Wrapper(self.value, parent=self)

    def branch_in(self, mode='sum', concat_dim=-1):
        vals = [self.value, self._parent.value]
        return Wrapper(vals, mode=mode, concat_dim=concat_dim, parent=self._parent._parent)


def wrap(var, mode='concat', concat_dim=-1):
    return Wrapper(var, mode=mode, concat_dim=concat_dim)
