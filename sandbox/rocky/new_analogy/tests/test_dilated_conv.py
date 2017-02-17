import numpy as np
from sandbox.rocky.th import tensor_utils
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from sandbox.rocky.th.core.modules import DilatedConv1d, DilatedConvNet, LayerSpec
from sandbox.rocky.th.ops import pad


def test_no_time_shift():
    x = np.arange(1, 11, dtype=np.float32)
    x = x.reshape((1, 1, 10))
    weight = np.reshape(np.array([0., 1.], dtype=np.float32), [1, 1, 2])  # 2, 1, 1])
    # x_padded = np.pad(x, [[0, 0], [0, 0], [2, 0]], 'constant')
    op = DilatedConv1d(
        in_channels=1, out_channels=1, kernel_size=2, dilation=1, bias=False, causal=True
    )
    op.weight = Parameter(torch.from_numpy(weight).float())
    result = tensor_utils.to_numpy(op(tensor_utils.variable(x)))
    np.testing.assert_allclose(result, x)


def test_pad():
    x = np.ones((3, 3, 3))
    out = tensor_utils.to_numpy(pad(tensor_utils.variable(x), [[0, 0], [0, 0], [1, 0]]))
    np.testing.assert_allclose(out[:, :, 0], 0)
    np.testing.assert_allclose(out[:, :, 1:], x)
    out = tensor_utils.to_numpy(pad(tensor_utils.variable(x), [[0, 0], [0, 0], [0, 1]]))
    np.testing.assert_allclose(out[:, :, -1], 0)
    np.testing.assert_allclose(out[:, :, :-1], x)


def test_causal():
    net = DilatedConvNet(
        in_channels=1,
        layer_specs=[
            LayerSpec(out_channels=1, kernel_size=2, dilation=1, causal=True),
            LayerSpec(out_channels=1, kernel_size=2, dilation=2, causal=True),
            LayerSpec(out_channels=1, kernel_size=2, dilation=4, causal=True),
            LayerSpec(out_channels=1, kernel_size=2, dilation=8, causal=True),
            LayerSpec(out_channels=1, kernel_size=2, dilation=16, causal=True),
            LayerSpec(out_channels=1, kernel_size=2, dilation=32, causal=True),
            # LayerSpec(out_channels=1, kernel_size=2, dilation=2, causal=True),
            # LayerSpec(out_channels=1, kernel_size=2, dilation=4, causal=True),
            # LayerSpec(out_channels=1, kernel_size=2, dilation=8, causal=True),
        ],
        # disable nonlinearity in case some variables are silenced out
        nonlinearity=None,
        output_nonlinearity=None,
    )
    x = np.arange(1, 101, dtype=np.float32)
    x = x.reshape((1, 1, -1))
    test_idx = 33
    output = tensor_utils.to_numpy(net(tensor_utils.variable(x)))[0, 0, test_idx]
    for idx in range(test_idx + 1, 100):
        x[0, 0, idx] += 1
        output_1 = tensor_utils.to_numpy(net(tensor_utils.variable(x)))[0, 0, test_idx]
        x[0, 0, idx] -= 1
        # changing future value should not affect past values
        np.testing.assert_allclose(output, output_1)
    for idx in range(test_idx + 1):
        x[0, 0, idx] += 1
        output_2 = tensor_utils.to_numpy(net(tensor_utils.variable(x)))[0, 0, test_idx]
        x[0, 0, idx] -= 1
        # should have some influence
        assert abs(output - output_2) > 1e-7


if __name__ == "__main__":
    test_pad()
    test_causal()
    # test_no_time_shift()
