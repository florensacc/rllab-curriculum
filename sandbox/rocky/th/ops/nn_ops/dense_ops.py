from sandbox.rocky.th.ops.context_ops import inc_scope, get_variable
from sandbox.rocky.th.ops.init_ops import uniform_initializer
from sandbox.rocky.th.ops.init_ops import zeros_initializer
from sandbox.rocky.th.ops.registry import register
import torch.nn.functional as F


@register
def dense(x, dim=None, bias=True, W_init=uniform_initializer(0.05), b_init=zeros_initializer(), name='dense', W=None,
          b=None, scope=None):
    batch_size, in_dim = x.size()
    if W is None or bias and b is None:
        assert dim is not None
        W_shape = (dim, in_dim)
        b_shape = (dim,)
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
    if b is None:
        return F.linear(x, W)
    return F.linear(x, W, b)


@register
def act(x, activation):
    if activation is None:
        return x
    return activation(x)
