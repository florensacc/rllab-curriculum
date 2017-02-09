from collections import namedtuple

import collections

from rllab.core.serializable import Serializable
import torch.nn.functional as F
from torch import nn
from sandbox.rocky.th.core.module_powered import ModulePowered
from sandbox.rocky.th import ops
from sandbox.rocky.th.ops import pad


def identity(x):
    return x


class MLP(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            nonlinearity=F.tanh,
            output_nonlinearity=identity,
            batch_norm=False,
            batch_norm_final=False,  # If batch_norm=True and batch_norm_final=False, won't batch norm the final layer
    ):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.output_nonlinearity = output_nonlinearity
        hs = []
        final_hidden_size = input_size
        for idx, (in_size, hidden_size) in enumerate(zip((input_size,) + tuple(hidden_sizes), hidden_sizes)):
            hs.append(nn.Linear(in_size, hidden_size))
            final_hidden_size = hidden_size
            setattr(self, "fc{}".format(idx), hs[-1])
            if batch_norm:
                hs.append(nn.BatchNorm1d(num_features=hidden_size))
                setattr(self, "bn{}".format(idx), hs[-1])
            hs.append(self.nonlinearity)
        hs.append(nn.Linear(final_hidden_size, output_size))
        setattr(self, "out", hs[-1])
        if batch_norm and batch_norm_final:
            hs.append(nn.BatchNorm1d(num_features=output_size))
            setattr(self, "bn_output", hs[-1])
        hs.append(self.output_nonlinearity)
        self.hs = hs

    def forward(self, x):
        for h in self.hs:
            x = h(x)
        return x


def define(typename, **fields):
    items = list(fields.items())
    # makes it deterministic between python runs.
    items = sorted(items)
    keys = [k for k, _ in items]
    values = [v for _, v in items]
    T = collections.namedtuple(typename, keys)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    prototype = T(*values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


class LayerSpec(collections.namedtuple(
    'LayerSpec',
    ['kernel_size', 'dilation', 'causal', 'stride', 'out_channels', 'structure']
)):
    """
    Structure can be one of 'resnet', 'plain'
    """
    __slots__ = ()

    def __new__(cls, out_channels, kernel_size=2, dilation=1, causal=False, stride=1, structure='resnet'):
        return super().__new__(
            cls,
            kernel_size=kernel_size, dilation=dilation, causal=causal, stride=stride,
            out_channels=out_channels, structure=structure
        )


class Linear_WN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

    pass


class DilatedConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, causal=False):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        self.causal = causal

    def forward(self, input):
        assert input.dim() == 3
        width = input.size(2)
        kernel_size, = self.kernel_size
        dilation, = self.dilation
        if self.causal:
            padding = (kernel_size - 1) * dilation
            padded_input = pad(input, [[0, 0], [0, 0], [padding, 0]])
        else:
            padding_total = (kernel_size - 1) * dilation
            padding_left = padding_total // 2
            padding_right = padding_total - padding_left
            padded_input = pad(input, [[0, 0], [0, 0], [padding_left, padding_right]])
        result = super().forward(padded_input)
        result = result[:, :, :width]
        return result


class DilatedConvNetSpec(
    collections.namedtuple('DilatedConvNetSpec', [
        'dilations',
        'n_channels',
        'kernel_size',
        'causal',
        'weight_norm'
    ])
):
    __slots__ = ()

    def __new__(cls, dilations, n_channels, kernel_size, causal, weight_norm=False):
        return super().__new__(
            cls,
            dilations=dilations, n_channels=n_channels, kernel_size=kernel_size, causal=causal, weight_norm=weight_norm
        )


class ResidualConvBlock(nn.Module):
    def __init__(self, n_channels, kernel_size, dilation, causal, nonlinearity=F.relu):
        super().__init__()
        self.conv_shorten = DilatedConv1d(
            in_channels=n_channels,
            out_channels=n_channels // 2,
            kernel_size=1,
            stride=1,
            dilation=1,
            causal=False,
        )
        self.conv_dilation = DilatedConv1d(
            in_channels=n_channels // 2,
            out_channels=n_channels // 2,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            causal=causal,
        )
        self.conv_widen = DilatedConv1d(
            in_channels=n_channels // 2,
            out_channels=n_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            causal=False,
        )
        self.nonlinearity = nonlinearity

    def forward(self, input):
        x = self.conv_shorten(self.nonlinearity(input))
        x = self.conv_dilation(self.nonlinearity(x))
        x = self.conv_widen(self.nonlinearity(x))
        return x + input


"""
what I would like to write
policy
with TH.scope("encoder"):
   var = wrap(demo_obs)
     .conv1d(size=1, dim=xx, act='relu')
     .conv1d(size=1, dim=xx, act='relu')
     .conv1d(size=1, dim=xx, act='relu')
     .conv1d(size=1, dim=xx, act='relu')
"""


class DilatedConvNet(nn.Module):
    def __init__(
            self,
            in_channels,
            spec,
            nonlinearity=F.relu,
            output_nonlinearity=F.relu,
    ):
        super().__init__()
        self.in_channels = in_channels
        # self.layer_specs = layer_specs
        hs = []
        layer = DilatedConv1d(
            in_channels=in_channels,
            out_channels=spec.n_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            causal=False,
        )
        setattr(self, "pre_conv", layer)
        hs.append(layer)
        for idx, dilation in enumerate(spec.dilations):
            res_block = ResidualConvBlock(
                n_channels=spec.n_channels,
                kernel_size=spec.kernel_size,
                dilation=dilation,
                causal=spec.causal,
                nonlinearity=nonlinearity,
            )
            setattr(self, "res{}".format(idx), res_block)
            hs.append(res_block)
        hs.append(output_nonlinearity)
        self.hs = hs

    def forward(self, x):
        for h in self.hs:
            x = h(x)
        return x
