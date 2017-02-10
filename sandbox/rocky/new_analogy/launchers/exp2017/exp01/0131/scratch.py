# import joblib
#
# from bin.tower_copter_policy import get_task_from_text
# from sandbox.rocky.new_analogy import fetch_utils
# from sandbox.rocky.new_analogy.policies import bytenet_ops
# from sandbox.rocky.s3 import resource_manager
# import tensorflow as tf
#
# task_id = "ab"
# n_configurations = 100
# task_id_arr = get_task_from_text(task_id)
# task_paths_resource = "fetch_analogy_paths/task_{}_trajs_{}.pkl".format(task_id, n_configurations)
# task_paths_filename = resource_manager.get_file(task_paths_resource)
# path = joblib.load(task_paths_filename)[2]
# # env = fetch_utils.fetch_env(horizon=500, height=5, task_id=task_id_arr)
# #
# #     # recomputing stages
# # path["env_infos"]["stage"] = fetch_utils.compute_stage(env, path["env_infos"]["site_xpos"])
# # # map to discrete-valued actions
# # # disc_actions = []
# # # for disc_idx in range(3):
# # #     cur_actions = path["actions"][:, disc_idx]
# # #     bins = np.asarray(fetch_utils.disc_intervals[disc_idx])
# # #     disc_actions.append(np.argmin(np.abs(cur_actions[:, None] - bins[None, :]), axis=1))
# # # disc_actions.append(np.cast['uint8'](p["actions"][:, -1] == 1))
# # # flat_actions = env_spec.action_space.flatten_n(np.asarray(disc_actions).T)
# # # p["actions"] = flat_actions
# #
# # is_success = lambda p: env.wrapped_env._is_success(p) and len(p["rewards"]) == horizon
#
# obs = path["observations"]
#
# import torch
# import torch.nn.functional as F
# import numpy as np
#
# obs = np.expand_dims(obs, 0)
#
# input_var = tf.placeholder(dtype=tf.float32, name="input", shape=obs.shape)
#
# tf.set_random_seed(0)
# dilation = 1
# filter_width = 3
#
# with tf.variable_scope("output_0"):
#     output_0_var = bytenet_ops.conv1d(
#         input_var, output_channels=32, filter_width=filter_width, name='dilated_conv')
# with tf.variable_scope("output_0", reuse=True):
#     o0w0_var = tf.get_variable("dilated_conv/w")
#     o0b0_var = tf.get_variable("dilated_conv/biases")
# with tf.variable_scope("output_1"):
#     output_1_var = bytenet_ops.dilated_conv1d(
#         input_var, output_channels=32, dilation=dilation, filter_width=filter_width, causal=True)
# with tf.variable_scope("output_1", reuse=True):
#     o1w0_var = tf.get_variable("dilated_conv/w")
#     o1b0_var = tf.get_variable("dilated_conv/biases")
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     output_0, output_1, o0w0, o0b0, o1w0, o1b0 = sess.run(
#         [output_0_var, output_1_var, o0w0_var, o0b0_var, o1w0_var, o1b0_var],
#         feed_dict={input_var: obs}
#     )
#     print("Dilated conv, dilation=2:", np.linalg.norm(output_0))
#     # output_1 = sess.run(output_1_var, feed_dict={input_var: obs})
#     print("Dilated conv, dilation=2, causal:", np.linalg.norm(output_1))
#
#     # with tf.variable_scope("output_0", reuse=True):
#     #     o0w0_var = tf.get_variable("dilated_conv/w")
#     #     o0b0_var = tf.get_variable("dilated_conv/biases")
#     # with tf.variable_scope("output_1", reuse=True):
#     #     o1w0_var = tf.get_variable("dilated_conv/w")
#     #     o1b0_var = tf.get_variable("dilated_conv/biases")
#
#     # o0w0, o0b0, o1w0, o1b0 = sess.run([o0w0_var, o0b0_var, o1w0_var, o1b0_var])
#
# from torch import autograd
#
# # obs: batch size * t * #channels
# obs = np.asarray(obs.transpose((0, 2, 1)), order='C')
# # o0w0: fwidth * inchannels * outchannels
# o0w0 = np.asarray(o0w0.transpose((2, 1, 0)), order='C')
#
# input = autograd.Variable(torch.from_numpy(obs).float())
# weights = autograd.Variable(torch.from_numpy(o0w0).float())
# bias = autograd.Variable(torch.from_numpy(o0b0).float())

# def dilated_conv1d(input, weights, bias, dilation=1):
#     """
#     :param input: assumes a minibatch * width * in_channels layout
#     :param weights: assumes a filter_width * in_channels * out_channels layout
#     :param bias: assumes of size out_channels
#     :param dilation: integer
#     """



# import torch.legacy.nn
#
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Function


#
# nn.Linear




# def pad_dim(x, dim, padding):
#     # assert padding >= 0, "Padding should be nonnegative, but got {}".format(padding)
#     if padding == 0:
#         return x
#     x_size = list(x.size())
#     x_size[dim] = abs(padding)
#     zeros = x.data.new(*x_size)
#     if padding < 0:
#         F.cat
#         return torch.cat([zeros, x], dim)
#     else:
#         return torch.cat([x, zeros], dim)




def test_no_time_shift():
    import numpy as np
    from sandbox.rocky.th import tensor_utils
    import torch
    from torch.autograd import Variable
    from torch.nn.parameter import Parameter
    x = np.arange(1, 11, dtype=np.float32)
    x = x.reshape((1, 1, 10))
    weight = np.reshape(np.array([1., 1], dtype=np.float32), [1, 1, 2])  # 2, 1, 1])
    # x_padded = np.pad(x, [[0, 0], [0, 0], [2, 0]], 'constant')
    op = CausalDilatedConv1d(
        in_channels=1, out_channels=1, kernel_size=2, dilation=1, bias=False
    )
    op.weight = Parameter(torch.from_numpy(weight).float())
    result = tensor_utils.to_numpy(op(tensor_utils.variable(x)))
    import ipdb;
    ipdb.set_trace()


if __name__ == "__main__":
    test_no_time_shift()







# input = torch.legacy.nn.Padding(dim=2, pad=(filter_width-1)*dilation//2).forward(input.data)
# input = torch.legacy.nn.Padding(dim=2, pad=-(filter_width-1)*dilation//2).forward(input)#.data)

# input = autograd.Variable(input)

# import ipdb; ipdb.set_trace()

# input =


# torch.nn.Conv1d

# g
# results = F.conv1d(
#     input,
#     weight=weights,
#     bias=bias,
#     dilation=dilation,
#     padding=0,#(filter_width - 1) * dilation // 2
# ).data.numpy()
# import ipdb;
#
# ipdb.set_trace()
# Dilated conv, dilation=2: 0.638469
# Dilated conv, dilation=2, causal: 1.48058

# ok! now it's torch's turn...
