import torch

from sandbox.rocky.th.ops import conversion_ops
from sandbox.rocky.th.ops.context_ops import get_variable, inc_scope
from sandbox.rocky.th.ops.init_ops import uniform_initializer, zeros_initializer
import torch.nn.functional as F
import numpy as np

from sandbox.rocky.th.ops.registry import register


class LSTMCell(object):
    def __init__(self, dim, W_init=uniform_initializer(0.05), b_init=zeros_initializer(), gate_act=F.sigmoid,
                 act=F.tanh, forget_bias=1.0):
        self.dim = dim
        self.W_init = W_init
        self.b_init = b_init
        self.gate_act = gate_act
        self.act = act
        self.forget_bias = forget_bias

    def init_state(self, batch_size):
        # the state consists of the hidden state and the cell state
        return conversion_ops.as_variable(np.zeros((batch_size, self.dim * 2)))

    def __call__(self, input, state):
        """
        Incoming gate:     i(t) = f_i(x(t) @ W_xi + h(t-1) @ W_hi + w_ci * c(t-1) + b_i)
        Forget gate:       f(t) = f_f(x(t) @ W_xf + h(t-1) @ W_hf + w_cf * c(t-1) + b_f)
        Cell gate:         c(t) = f(t) * c(t - 1) + i(t) * f_c(x(t) @ W_xc + h(t-1) @ W_hc + b_c)
        Out gate:          o(t) = f_o(x(t) @ W_xo + h(t-1) W_ho + w_co * c(t) + b_o)
        New hidden state:  h(t) = o(t) * f_h(c(t))

        Typically, f_i, f_f, and f_o are chosen to be sigmoid, and f_c, f_h are chosen to be tanh
        """

        input_dim = input.size(1)
        hprev = state[:, :self.dim]
        cprev = state[:, self.dim:]
        # weights for the incoming gate
        W_x = get_variable(name="W_x", shape=(4 * self.dim, input_dim), initializer=self.W_init,
                           trainable=True, regularizable=True)
        W_h = get_variable(name="W_h", shape=(4 * self.dim, self.dim), initializer=self.W_init,
                           trainable=True, regularizable=True)
        b = get_variable(name="b", shape=(4 * self.dim,), initializer=self.b_init,
                         trainable=True, regularizable=False)

        gates = F.linear(input, W_x, b) + F.linear(hprev, W_h)

        # compute the sum of terms, before applying more transformations
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        in_gate = self.gate_act(in_gate)
        forget_gate = self.gate_act(forget_gate + self.forget_bias)
        cell_gate = self.act(cell_gate)
        out_gate = self.gate_act(out_gate)

        cy = forget_gate * cprev + in_gate * cell_gate
        hy = out_gate * self.act(cy)

        return hy, torch.cat([hy, cy], 1)


@register
def rnn(x, cell, time_dim=1, reversed=False, name='rnn'):
    outputs = []
    if reversed:
        t_range = range(x.size(time_dim) - 1, -1, -1)
    else:
        t_range = range(0, x.size(time_dim))
    batch_size = x.size(0)
    with inc_scope(name):
        state = cell.init_state(batch_size)
        for t in t_range:
            xt = x.select(time_dim, t)
            output, state = cell(xt, state)
            outputs.append(output.unsqueeze(time_dim))
        if reversed:
            outputs = outputs[::-1]
        return torch.cat(outputs, time_dim)


def birnn(x, cell_fwd, cell_bwd, time_dim=2, name='birnn'):
    fwd = rnn(x, cell_fwd, time_dim=time_dim, name=name + '_fwd')
    bwd = rnn(x, cell_bwd, time_dim=time_dim, reversed=True, name=name + '_bwd')
    return torch.cat([fwd, bwd], time_dim)
