# Phases
from collections import OrderedDict
from contextlib import contextmanager
import numpy as np
import torch
from torch.nn import Parameter
import os

from sandbox.rocky.th.ops import conversion_ops

TRAIN = 0
TEST = 1
INIT = 2

_variables = OrderedDict()
_tags = OrderedDict()
_scope = []
_scope_str = ""
# store the increments within each scope
_scope_counter = OrderedDict()
_defaults = OrderedDict()
_phase = TRAIN


def scope(name):
    if isinstance(name, Scope):
        return name
    return Scope(name)


@contextmanager
def phase(new_phase):
    global _phase
    old_phase = _phase
    _phase = new_phase
    yield
    _phase = old_phase


def get_phase():
    return _phase


class Scope(object):
    def __init__(self, name, scope_list=None):
        if scope_list is None:
            self._scope_list = list(_scope) + [name]
        else:
            self._scope_list = scope_list
        self._old_scope = None

    def __enter__(self):
        global _scope
        global _scope_str
        self._old_scope = list(_scope)
        _scope = list(self._scope_list)
        _scope_str = "/".join(_scope)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _scope
        global _scope_str
        for key in _scope_counter:
            if key.startswith(_scope_str):
                _scope_counter[key] = -1
        _scope = list(self._old_scope)
        self._old_scope = None
        _scope_str = "/".join(_scope)


# @contextmanager
def inc_scope(name):
    if isinstance(name, Scope):
        return Scope(name=None, scope_list=list(name._scope_list))
    if _scope_str not in _scope_counter:
        _scope_counter[_scope_str] = -1
    _scope_counter[_scope_str] += 1
    return scope(name + ":" + str(_scope_counter[_scope_str]))


def get_variable(name, shape=None, initializer=None, dtype='float', trainable=True,
                 regularizable=True):
    full_name = "/".join(_scope + [name])
    if full_name not in _variables:
        assert initializer is not None
        if isinstance(initializer, np.ndarray):
            if shape is not None:
                assert initializer.shape == tuple(shape)
            init_val = initializer
        else:
            assert shape is not None
            init_val = initializer(shape)
        var = conversion_ops.variable(
            init_val, dtype=dtype, requires_grad=trainable)
        var = Parameter(var.data, requires_grad=True)
        var._ops_full_name = full_name
        _variables[full_name] = var
        _tags[full_name] = dict(trainable=trainable,
                                regularizable=regularizable)
    return _variables[full_name]


def scoped_variables(scope_name):
    ret = []
    with scope(scope_name):
        for key, var in _variables.items():
            if key.startswith(_scope_str):
                ret.append(var)
    return ret


def get_tags(var):
    return _tags[var._ops_full_name]


def get_scope_str():
    return _scope_str


def is_trainable(var):
    return get_tags(var)['trainable'] == True


def is_regularizable(var):
    return get_tags(var)['regularizable'] == True


# mainly used for testing
def reset():
    global _phase
    global _scope_str
    _variables.clear()
    _tags.clear()
    _defaults.clear()
    _scope_counter.clear()
    _scope.clear()
    _scope_str = ""
    _phase = TRAIN
