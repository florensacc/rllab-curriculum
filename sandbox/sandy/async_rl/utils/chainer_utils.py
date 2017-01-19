import multiprocessing as mp
import os
import chainer
import numpy as np

def extract_link_params(link):
    """
    For each param in the link, deep copy its data as a vectorized array to a shared memory address.
    """
    assert isinstance(link, chainer.Link)
    shared_arrays = {}
    for param_name, param in link.namedparams():
        shared_arrays[param_name] = mp.RawArray('f', param.data.ravel())
    return shared_arrays


def set_link_params(target_link, params): # model params
    """
    Create a shallow copy (a) of model parameters (b).
    Args:
      a (chainer.Link): link whose params are to be replaced
      b (dict): dict that consists of (param_name, multiprocessing.Array)
    """
    assert isinstance(target_link, chainer.Link)
    for param_name, param in target_link.namedparams():
        if param_name in params:
            shared_param = params[param_name]
            param.data = np.frombuffer(
                shared_param, dtype=param.data.dtype).reshape(param.data.shape)


def set_optimizer_params(target_optimizer, params): # optimizer state
    """
    Create a shallow copy (a) of an optimizer state (b).
    """
    assert isinstance(target_optimizer, chainer.Optimizer)
    assert hasattr(target_optimizer, 'target'), 'Optimizer.setup must be called first'
    for state_name, shared_state in params.items(): # state includes params ...
        for param_name, param in shared_state.items():
            old_param = target_optimizer._states[state_name][param_name]
            target_optimizer._states[state_name][param_name] = np.frombuffer(
                param,
                dtype=old_param.dtype).reshape(old_param.shape)

def extract_optimizer_params(optimizer):
    assert isinstance(optimizer, chainer.Optimizer)
    assert hasattr(optimizer, 'target'), 'Optimizer.setup must be called first'
    shared_arrays = {}
    for state_name, state in optimizer._states.items():
        shared_arrays[state_name] = {}
        for param_name, param in state.items():
            shared_arrays[state_name][
                param_name] = mp.RawArray('f', param.ravel())
    return shared_arrays

def copy_link_param(target_link, source_link):
    """Copy parameters of a link to another link.
    """
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        target_params[param_name].data[:] = param.data


def copy_link_grad(target_link, source_link):
    """Copy gradients of a link to another link.
    """
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        target_params[param_name].grad[:] = param.grad
