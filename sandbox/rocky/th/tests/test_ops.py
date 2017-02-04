import torch
import torch.nn
import numpy as np
from sandbox.rocky.th import ops


def test_transpose_gradient():
    x = ops.as_variable(np.random.randn(4, 6, 5))
    x = torch.transpose(x, 1, 2)
    conv = torch.nn.Conv1d(in_channels=5, out_channels=10, kernel_size=2)
    result = torch.sum(conv(x))
    result.backward()


def test_conv1d_generation():
    for size in [2, 3, 4, 5]:
        ops.reset()
        np.random.seed(0)
        batch_size = 1  # 0
        dim = 32
        horizon = 10  # 2  # 1#20
        x = ops.as_variable(np.random.randn(batch_size, dim, horizon))

        with ops.scope("test"):
            result_1 = x
            for rate in [1, 3, 5, 7, 8]:
                result_1 = ops.conv1d_wn(result_1, dim=dim, size=size, causal=True,
                                         rate=rate)
            result_1 = ops.to_numpy(result_1)
        gen_queues = dict()
        conv_op = ops.mk_conv1d_generation(ops.conv1d_wn, scope_name="conv1d_wn", gen_queues=gen_queues)
        for idx in range(horizon):
            x_i = x[:, :, idx:idx + 1]
            with ops.scope("test"):
                result_i = x_i
                for rate in [1, 3, 5, 7, 8]:
                    result_i = conv_op(result_i, dim=dim, size=size, causal=True, rate=rate)
                result_i = ops.to_numpy(result_i)
                max_diff = np.max(np.abs(result_1[:, :, idx] - result_i[:, :, 0]))
                assert max_diff < 1e-5, \
                    "Conv op results differed (by max {}) on step {} (size {}, rate {})".format(max_diff, idx, size,
                                                                                                rate)


def test_lstm():
    ops.reset()
    np.random.seed(0)
    batch_size = 5
    seq_len = 10
    dim = 32
    hidden_size = 16
    seq = np.random.randn(batch_size, seq_len, dim)
    with ops.scope("test"):
        cell = ops.LSTMCell(dim=hidden_size)
        result = ops.to_numpy(ops.wrap(seq).rnn(cell).value)
    # now, try to reproduce the same result using the pytorch builtin LSTM (cross check!), using the same
    # weight parameters
    lstm = torch.nn.LSTM(input_size=dim, hidden_size=hidden_size, batch_first=True)
    weight_ih_l0 = lstm.weight_ih_l0
    weight_hh_l0 = lstm.weight_hh_l0
    bias_ih_l0 = lstm.bias_ih_l0
    bias_hh_l0 = lstm.bias_hh_l0

    W_x_ifco = ops.get_variable('test/rnn:0/W_x')
    W_h_ifco = ops.get_variable('test/rnn:0/W_h')
    b_ifco = ops.get_variable('test/rnn:0/b')

    weight_ih_l0.data.copy_(W_x_ifco.data)
    weight_hh_l0.data.copy_(W_h_ifco.data)
    bias_ih_l0.data.copy_(b_ifco.data)
    bias_ih_l0.data[hidden_size:2 * hidden_size] += cell.forget_bias
    bias_hh_l0.data.fill_(0)

    result_2 = ops.to_numpy(lstm(
        ops.as_variable(seq),
        (
            ops.as_variable(torch.zeros((1, batch_size, hidden_size))),
            ops.as_variable(torch.zeros((1, batch_size, hidden_size)))
        )
    )[0])

    for t in range(seq_len):
        # print("Checking t={}, max diff: {}".format(t, np.max(np.abs(result[:,t,:] - result_2[:,t,:]))))
        np.testing.assert_allclose(result[:, t, :], result_2[:, t, :], atol=1e-8, rtol=1)


def test_scope_reuse():
    import numpy as np
    a = np.random.randn(16, 32, 100)
    ops.reset()
    with ops.scope("aa"):
        b = ops.to_numpy(ops.wrap(a).conv1d(dim=64, size=3).value)
    with ops.scope("aa"):
        c = ops.to_numpy(ops.wrap(a).conv1d(dim=64, size=3).value)
    np.testing.assert_allclose(b, c)


if __name__ == "__main__":
    test_lstm()
    test_conv1d_generation()
    test_transpose_gradient()
    test_scope_reuse()
