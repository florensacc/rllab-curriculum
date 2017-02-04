from sandbox.rocky.th.ops import functional_ops
from sandbox.rocky.th.ops.context_ops import inc_scope, get_variable, get_phase, INIT, get_scope_str
from sandbox.rocky.th.ops.functional_ops import l2_normalize
from sandbox.rocky.th.ops.init_ops import uniform_initializer, zeros_initializer, ones_initializer
from sandbox.rocky.th.ops.registry import register
from sandbox.rocky.th.ops import conversion_ops
import torch.nn.functional as F
import numpy as np


def _patched_view_4d(*tensors):
    output = []
    for t in tensors:
        assert t.dim() == 3
        size = list(t.size())
        size.insert(2, 1)
        output += [t.contiguous().view(*size)]
    return output


import torch.nn._functions.conv

torch.nn._functions.conv._view4d = _patched_view_4d


# dim: #output channels
# size: filter size
# stride > 1 not supported
@register
def conv1d(x, dim, size, bias=True, pad='same', rate=1, causal=False,
           W_init=uniform_initializer(0.05), b_init=zeros_initializer(),
           name='conv1d', W=None, b=None, scope=None):
    batch_size, in_channels, width = x.size()
    out_channels = dim
    kernel_size = size
    W_shape = (out_channels, in_channels, kernel_size)
    b_shape = (out_channels,)
    if W is None or bias and b is None:
        with inc_scope(scope or name):
            if W is None:
                W = get_variable(
                    name="W", shape=W_shape, initializer=W_init, trainable=True, regularizable=True)
            if bias:
                if b is None:
                    b = get_variable(
                        name="b", shape=b_shape, initializer=b_init, trainable=True, regularizable=False)
            else:
                b = None
    if pad == 'same':
        if causal:
            pad_left = (size - 1) * rate
            pad_right = 0
        else:
            padding = (size - 1) * rate
            pad_left = padding // 2
            pad_right = padding - pad_left
    elif pad == 'valid':
        assert not causal
        pad_left = 0
        pad_right = 0
    else:
        raise NotImplementedError
    if pad_left > 0 or pad_right > 0:
        x_padded = functional_ops.pad(x, [[0, 0], [0, 0], [pad_left, pad_right]])
    else:
        x_padded = x
    y = F.conv1d(x_padded, weight=W, bias=b,
                 stride=1, padding=0, dilation=rate)
    if pad == 'same' and (pad_left > 0 or pad_right > 0):
        y = y[:, :, :width]
    return y


@register
def conv1d_wn(x, dim, size, bias=True, pad='same', rate=1, causal=False, name='conv1d_wn',
              init_scale=1., W_init=uniform_initializer(0.05), b_init=None, scope=None):
    assert bias
    assert b_init is None
    with inc_scope(scope or name):
        batch_size, in_channels, width = x.size()
        out_channels = dim
        kernel_size = size
        W_shape = (out_channels, in_channels, kernel_size)
        b_shape = (out_channels,)

        if get_phase() == INIT:
            V = get_variable(name="V", shape=W_shape, initializer=W_init,
                             trainable=True, regularizable=False)
            # normalize each feature map in V to standard L2 norm
            V_norm = l2_normalize(V, dims=[1, 2])
            x_init = conv1d(x, dim=dim, size=size, bias=False,
                            W=V_norm, pad=pad, rate=rate, causal=causal)
            x_init_val = conversion_ops.to_numpy(x_init)
            mean_init = np.mean(x_init_val, axis=(0, 2), keepdims=True)
            std_init = np.std(x_init_val, axis=(0, 2), keepdims=True)
            scale_init = init_scale / (std_init + 1e-8)
            g = get_variable(name="g", shape=b_shape, initializer=scale_init.reshape(b_shape), trainable=True,
                             regularizable=True)
            # in case g is already constructed, re-initialize
            b = get_variable(name="b", shape=b_shape, initializer=(-mean_init * scale_init).reshape(b_shape),
                             trainable=True, regularizable=False)
            x_init = g.view(1, -1, 1).expand_as(x_init) * \
                     x_init + b.view(1, -1, 1).expand_as(x_init)
            return x_init
        else:
            V = get_variable(name="V", shape=W_shape, initializer=W_init,
                             trainable=True, regularizable=False)
            g = get_variable(name="g", shape=b_shape, initializer=ones_initializer(), trainable=True,
                             regularizable=True)
            b = get_variable(name="b", shape=b_shape, initializer=zeros_initializer(), trainable=True,
                             regularizable=False)
            V_norm = l2_normalize(V, dims=[1, 2])
            W = g.view(-1, 1, 1).expand_as(V_norm) * V_norm
            return conv1d(x, dim=dim, size=size, bias=bias, pad=pad, rate=rate, causal=causal, W=W, b=b)


def mk_conv1d_generation(conv1d_op, scope_name, gen_queues):
    """
    This method generates an op in place of the provided 1D causal convolution op, which caches the computed results
    """

    def conv1d_generation(x, dim, size, rate=1, causal=False, **kwargs):
        with inc_scope(scope_name) as scope:
            assert x.size(2) == 1
            batch_size = x.size(0)
            if size == 1:
                # No need for a queue
                return conv1d_op(x, dim=dim, size=size, rate=rate, causal=causal, scope=scope, **kwargs)
            assert causal
            in_channels = x.size(1)
            scope_str = get_scope_str()
            if scope_str not in gen_queues:
                gen_queues[scope_str] = [
                    [np.zeros((batch_size, in_channels)) for _ in range(rate)]
                    for _ in range(size - 1)
                    ]
            queues = gen_queues[scope_str]
            # pop the last elements
            popped = [q.pop(0) for q in queues]
            for idx, popped_i in enumerate(popped[1:]):
                queues[idx].append(popped_i)
            queues[-1].append(conversion_ops.to_numpy(x)[:, :, 0])
            popped = conversion_ops.as_variable(
                np.asarray(popped).transpose((1, 2, 0)))
            joint_x = torch.cat([popped, x], 2)
            prod = conv1d_op(joint_x, dim=dim, size=size,
                             rate=1, causal=False, pad='valid', scope=scope, **kwargs)
            assert prod.size(2) == 1
            return prod

    return conv1d_generation
