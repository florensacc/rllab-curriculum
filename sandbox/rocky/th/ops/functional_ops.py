import torch
from torch.autograd import Function


class Pad(Function):
    def __init__(self, spec, value=0):
        super().__init__()
        self.spec = spec
        self.value = value

    def forward(self, x):
        shape = tuple(x.size())
        padded_shape = tuple(l + d + r for d, (l, r) in zip(shape, self.spec))
        slice_index = tuple(slice(l, d + l)
                            for d, (l, r) in zip(shape, self.spec))
        new_tensor = x.new(*padded_shape)
        new_tensor.fill_(self.value)
        new_tensor[slice_index] = x
        return new_tensor

    def backward(self, grad_output):
        padded_shape = tuple(grad_output.size())
        shape = tuple(d - l - r for d, (l, r) in zip(padded_shape, self.spec))
        slice_index = tuple(slice(l, d + l)
                            for d, (l, r) in zip(shape, self.spec))
        return grad_output[slice_index]


def pad(x, spec):
    if all([l == 0 and r == 0 for l, r in spec]):
        return x
    return Pad(spec)(x)


def l2_normalize(x, dims, eps=1e-12):
    if isinstance(dims, int):
        dims = [dims]
    sumsqr = x ** 2
    for dim in dims:
        sumsqr = torch.sum(sumsqr, dim)
    return x / torch.sqrt(sumsqr.expand_as(x) + eps)
