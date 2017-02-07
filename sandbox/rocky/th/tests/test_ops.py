import gc
import pyprind
import torch
import torch.nn
from torch import autograd
import numpy as np
import time
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
        np.testing.assert_allclose(result[:, t, :], result_2[:, t, :], atol=1e-8, rtol=0)


def test_scope_reuse():
    import numpy as np
    a = np.random.randn(16, 32, 100)
    ops.reset()
    with ops.scope("aa"):
        b = ops.to_numpy(ops.wrap(a).conv1d(dim=64, size=3).value)
    with ops.scope("aa"):
        c = ops.to_numpy(ops.wrap(a).conv1d(dim=64, size=3).value)
    np.testing.assert_allclose(b, c)


def test_dense_wn():
    # how to test this hmm
    batch_size = 16
    in_dim = 64
    out_dim = 32
    ops.reset()
    x = np.random.randn(batch_size, in_dim)
    with ops.phase(ops.INIT):
        with ops.scope("test"):
            init_out = ops.to_numpy(ops.wrap(x).dense_wn(out_dim).value)
    # it should be the case that after initialization, the output has zero mean and unit variance
    np.testing.assert_allclose(init_out.std(axis=0), np.ones(out_dim), rtol=0, atol=1e-6)
    np.testing.assert_allclose(init_out.mean(axis=0), np.zeros(out_dim), rtol=0, atol=1e-6)
    with ops.phase(ops.TRAIN):
        with ops.scope("test"):
            train_out = ops.to_numpy(ops.wrap(x).dense_wn(out_dim).value)
    # In training mode, immediately passing in the same input should still have normalized outputs
    np.testing.assert_allclose(train_out.std(axis=0), np.ones(out_dim), rtol=0, atol=1e-6)
    np.testing.assert_allclose(train_out.mean(axis=0), np.zeros(out_dim), rtol=0, atol=1e-6)
    # But not for a different input
    y = np.random.randn(batch_size, in_dim)
    with ops.phase(ops.TRAIN):
        with ops.scope("test"):
            train_out = ops.to_numpy(ops.wrap(y).dense_wn(out_dim).value)
    assert not np.allclose(train_out.std(axis=0), np.ones(out_dim), rtol=0, atol=1e-6)
    assert not np.allclose(train_out.mean(axis=0), np.zeros(out_dim), rtol=0, atol=1e-6)


def test_lstm_speed():
    batch_size = 16
    T = 500
    in_dim = 256
    lstm_dim = 64
    ops.reset()
    x = np.random.randn(batch_size, T, in_dim)

    def test_tensorflow_basic_lstm():
        # construct tensorflow lstm
        import tensorflow as tf
        x_var = tf.placeholder(dtype=tf.float32, shape=(batch_size, T, in_dim))
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_dim)
        state = cell.zero_state(batch_size, dtype=tf.float32)
        outputs = []
        with tf.variable_scope("test"):
            for t in range(T):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                output, state = cell(x_var[:, t, :], state)
                outputs.append(output)
        output = tf.reduce_sum(outputs[-1])
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            # first run, not counting time
            sess.run(output, feed_dict={x_var: x})
            start_time = time.time()
            for _ in pyprind.prog_bar(range(100)):
                sess.run(outputs, feed_dict={x_var: x})
            end_time = time.time()
            print("tensorflow basic lstm took {}s".format(end_time - start_time))

    def test_tensorflow_dynamic_lstm():
        # construct tensorflow lstm
        import tensorflow as tf
        x_var = tf.placeholder(dtype=tf.float32, shape=(batch_size, T, in_dim))
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_dim)
        # state = cell.zero_state(batch_size, dtype=tf.float32)

        outputs, _ = tf.nn.dynamic_rnn(cell, x_var, dtype=tf.float32)
        # outputs = []
        # with tf.variable_scope("test"):
        #     for t in range(T):
        #         if t > 0:
        #             tf.get_variable_scope().reuse_variables()
        #         output, state = cell(x_var[:, t, :], state)
        #         outputs.append(output)
        output = tf.reduce_sum(outputs[-1])
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            # first run, not counting time
            sess.run(output, feed_dict={x_var: x})
            start_time = time.time()
            for _ in pyprind.prog_bar(range(100)):
                sess.run(output, feed_dict={x_var: x})
            end_time = time.time()
            print("tensorflow dynamic lstm took {}s".format(end_time - start_time))

    def test_tensorflow_cudnn_lstm():
        # construct tensorflow lstm
        import tensorflow as tf
        x_var = tf.placeholder(dtype=tf.float32, shape=(batch_size, T, in_dim))
        x_var_t = tf.transpose(x_var, (1, 0, 2))
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(1, lstm_dim, in_dim)
        input_h = tf.ones((1, batch_size, lstm_dim))
        input_c = tf.ones((1, batch_size, lstm_dim))
        params_size_t = lstm.params_size()
        params = tf.Variable(
            tf.ones([params_size_t]), validate_shape=False)
        output, output_h, output_c = lstm(
            is_training=False,
            input_data=x_var_t,
            input_h=input_h,
            input_c=input_c,
            params=params,
        )
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            # first run, not counting time
            sess.run(output_h, feed_dict={x_var: x})
            start_time = time.time()
            for _ in pyprind.prog_bar(range(100)):
                sess.run(output_h, feed_dict={x_var: x})
            end_time = time.time()
            print("tensorflow cudnn lstm took {}s".format(end_time - start_time))

    # # initialize cuda

    def test_torch_ops_lstm():
        # only evaluate performance of forward pass
        x_wrapped = ops.wrap(x, volatile=True)
        x_wrapped.rnn(ops.LSTMCell(lstm_dim))
        start_time = time.time()
        for _ in pyprind.prog_bar(range(100)):
            with ops.scope("test"):
                x_wrapped.rnn(ops.LSTMCell(lstm_dim))
        end_time = time.time()
        print("ops lstm took {}s".format(end_time - start_time))

    def test_torch_lstm():
        # construct a pytorch builtin LSTM
        lstm = torch.nn.LSTM(input_size=in_dim, hidden_size=lstm_dim, num_layers=1, batch_first=True)
        lstm.cuda()
        h0 = ops.as_variable(np.zeros((1, batch_size, lstm_dim)), volatile=True)
        c0 = ops.as_variable(np.zeros((1, batch_size, lstm_dim)), volatile=True)
        x_var = ops.as_variable(x, volatile=True)
        # dry run, to eliminate overhead of initializing variables
        lstm(x_var, (h0, c0))
        start_time = time.time()
        for _ in pyprind.prog_bar(range(100)):
            lstm(x_var, (h0, c0))
        end_time = time.time()
        print("torch lstm took {}s".format(end_time - start_time))

    def test_torch_lstm_cell():
        # construct a pytorch builtin LSTM cell
        cell = torch.nn.LSTMCell(input_size=in_dim, hidden_size=lstm_dim)
        cell.cuda()
        h0 = ops.as_variable(np.zeros((batch_size, lstm_dim)), volatile=True)
        c0 = ops.as_variable(np.zeros((batch_size, lstm_dim)), volatile=True)
        start_time = time.time()
        x_var = ops.as_variable(x, volatile=True)
        for _ in pyprind.prog_bar(range(100)):
            hx = (h0, c0)
            for t in range(T):
                # cell(x, )
                hx = cell(x_var[:, t, :], hx)
        end_time = time.time()
        print("torch lstm cell took {}s".format(end_time - start_time))

    # test_tensorflow_basic_lstm() # ~ 15 seconds
    # test_tensorflow_cudnn_lstm() # ~ 1.75 seconds
    # test_tensorflow_dynamic_lstm() # ~ 11.37 seconds
    test_torch_ops_lstm()  # ~ 22 seconds
    test_torch_lstm()  # ~ 2.09 seconds
    test_torch_lstm_cell()  # ~ 18 seconds


if __name__ == "__main__":
    # test_dense_wn()
    # test_lstm()
    # test_conv1d_generation()
    # test_transpose_gradient()
    # test_scope_reuse()
    test_lstm_speed()
