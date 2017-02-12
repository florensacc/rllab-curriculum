import torch
import numpy as np

from sandbox.rocky.th.ops.base_ops import is_cuda


def variable(obj, dtype='float', volatile=None, requires_grad=None):
    if isinstance(obj, torch.autograd.Variable):
        return variable_from_tensor(obj.data, dtype=dtype, volatile=volatile, requires_grad=requires_grad)
    elif isinstance(obj, torch._TensorBase):
        return variable_from_tensor(obj, dtype=dtype, volatile=volatile, requires_grad=requires_grad)
    elif isinstance(obj, np.ndarray):
        return variable_from_numpy(obj, dtype=dtype, volatile=volatile, requires_grad=requires_grad)
    else:
        raise NotImplementedError


def as_variable(obj, dtype='float', volatile=None, requires_grad=None):
    if isinstance(obj, torch.autograd.Variable):
        return obj
    return variable(obj, dtype=dtype, volatile=volatile, requires_grad=requires_grad)


def variable_from_tensor(data, dtype='float', volatile=None, requires_grad=None):
    if dtype == 'float':
        data = data.float()
    elif dtype == 'int':
        data = data.int()
    elif dtype == 'long':
        data = data.long()
    elif dtype == 'byte':
        data = data.byte()
    else:
        raise NotImplementedError
    if is_cuda():
        data = data.cuda()
    kwargs = dict()
    if volatile is not None:
        kwargs["volatile"] = volatile
        if volatile:
            requires_grad = False
    if requires_grad is not None:
        kwargs["requires_grad"] = requires_grad
    return torch.autograd.Variable(data, **kwargs)


def variable_from_numpy(arr, dtype='float', volatile=None, requires_grad=None):
    arr = np.asarray(arr, order='C')
    data = torch.from_numpy(arr)
    return variable_from_tensor(data, dtype=dtype, volatile=volatile, requires_grad=requires_grad)


def to_numpy(var):
    if isinstance(var, torch._TensorBase):
        data = var
    elif isinstance(var, torch.autograd.Variable):
        data = var.data
    else:
        raise NotImplementedError
    if is_cuda():
        data = data.cpu()
    return data.numpy()
