from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import random
import scipy.sparse
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
from rllab.core.lasagne_layers import OpLayer
import theano
import theano.tensor as TT
import pyprind
import os.path
import scipy.io as sio
from rllab.misc import special
from rllab.misc import logger
from rllab.misc import ext
from rllab.optimizers.minibatch_dataset import BatchDataset
from rllab import config
from rllab.core.network import wrapped_conv

floatX = theano.config.floatX

# 0: left top
# 1: top
# 2: right top
# 3: right
# 4: right bottom
# 5: bottom
# 6: left bottom
# 7: left

ACTION_MAP = np.array([
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
    [1, 0],
    [1, -1],
    [0, -1],
])


def logical_or(*args):
    if len(args) == 1:
        return args[0]
    ret = args[0]
    for x in args[1:]:
        ret = np.logical_or(ret, x)
    return ret


def _insert_circle(canvas, x, y, radius):
    """
    Add a circle to the canvas, modify the memory content in-place
    :param x: center x
    :param y: center y
    :param radius: radius of circle
    :return:
    """
    nrow, ncol = canvas.shape
    xx, yy = np.mgrid[:nrow, :ncol]
    circle = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    canvas[circle] = 1


def _insert_rect(canvas, x, y, width, height):
    """
    Add a rectangle to the canvas, modify the memory content in-place
    :param x: left top x
    :param y: left top y
    :param width: width of rect
    :param height: height of rect
    :return:
    """
    nrow, ncol = canvas.shape
    xx, yy = np.mgrid[:nrow, :ncol]
    rect = np.logical_and(
        np.logical_and(xx >= x, yy >= y),
        np.logical_and(xx <= x + height, yy <= y + width),
    )
    canvas[rect] = 1


def new_map(shape, n_obstacles):
    """
    Generate a random map of the specified size and number of obstacles
    """
    map = np.zeros(shape)
    nrow, ncol = shape

    obs_types = ['circ', 'rect']
    max_obstacle_size = max(nrow, ncol) / 4

    for _ in xrange(n_obstacles):
        rand_type = random.choice(obs_types)
        if rand_type == 'circ':
            rand_radius = np.random.randint(low=1, high=max_obstacle_size + 1)
            randx = np.random.randint(nrow)
            randy = np.random.randint(ncol)
            _insert_circle(map, randx, randy, rand_radius)
        elif rand_type == 'rect':
            rand_hgt = np.random.randint(low=1, high=max_obstacle_size + 1)
            rand_wid = np.random.randint(low=1, high=max_obstacle_size + 1)
            randx = np.random.randint(nrow)
            randy = np.random.randint(ncol)
            _insert_rect(map, randx, randy, rand_wid, rand_hgt)
        else:
            raise NotImplementedError

    return map


def to_sparse_adj_graph(map):
    """
    Assume map is represented as: 0 -- empty, 1 -- blocked. Convert the map to a sparse graph representation
    """

    ids = np.where(1 - map.flatten())[0]
    nrow, ncol = map.shape
    xs, ys = ids / ncol, ids % ncol

    row_ids = []
    col_ids = []

    # horizontal edge: only present if ids not the first
    h_ids = np.logical_and(ys != 0, np.logical_not(map[xs, ys - 1]))
    hx, hy = xs[h_ids], ys[h_ids]
    from_hids = hx * ncol + hy
    to_hx, to_hy = hx, hy - 1
    to_hids = to_hx * ncol + to_hy
    row_ids.append(from_hids)
    row_ids.append(to_hids)
    col_ids.append(to_hids)
    col_ids.append(from_hids)

    # vertical edge
    v_ids = np.logical_and(xs != 0, np.logical_not(map[xs - 1, ys]))
    vx, vy = xs[v_ids], ys[v_ids]
    from_vids = vx * ncol + vy
    to_vx, to_vy = vx - 1, vy
    to_vids = to_vx * ncol + to_vy
    row_ids.append(from_vids)
    row_ids.append(to_vids)
    col_ids.append(to_vids)
    col_ids.append(from_vids)

    # major diagonal edge
    md_ids = np.logical_and(np.logical_and(xs != 0, ys != 0), np.logical_not(map[xs - 1, ys - 1]))
    mdx, mdy = xs[md_ids], ys[md_ids]
    from_mdids = mdx * ncol + mdy
    to_mdx, to_mdy = mdx - 1, mdy - 1
    to_mdids = to_mdx * ncol + to_mdy
    row_ids.append(from_mdids)
    row_ids.append(to_mdids)
    col_ids.append(to_mdids)
    col_ids.append(from_mdids)

    # sub diagonal edge
    sd_ids = np.logical_and(np.logical_and(xs != 0, ys != ncol - 1), np.logical_not(map[xs - 1, (ys + 1) % ncol]))
    sdx, sdy = xs[sd_ids], ys[sd_ids]
    from_sdids = sdx * ncol + sdy
    to_sdx, to_sdy = sdx - 1, sdy + 1
    to_sdids = to_sdx * ncol + to_sdy
    row_ids.append(from_sdids)
    row_ids.append(to_sdids)
    col_ids.append(to_sdids)
    col_ids.append(from_sdids)

    # Start adding these edges

    row_ids = np.concatenate(row_ids)  # [from_hids, to_hids, from_vids, to_vids])
    col_ids = np.concatenate(col_ids)  # [to_hids, from_hids, to_vids, from_vids])

    edge_vals = np.concatenate([
        np.ones(len(from_hids) * 2),
        np.ones(len(from_vids) * 2) - 1e-5,
        np.ones(len(from_mdids) * 2) * np.sqrt(2) + 1e-5,
        np.ones(len(from_sdids) * 2) * np.sqrt(2) + 2e-5,
    ])

    return scipy.sparse.csr_matrix((edge_vals, (row_ids, col_ids)), shape=(nrow * ncol, nrow * ncol))


def gen_demos(shape, max_n_obstacles, n_maps):
    """
    Generate demonstration trajectories
    :param shape: Shape of the map
    :param max_n_obstacles: Maximum number of obstacles
    :param n_maps: Number of random maps
    """

    traj_states = []
    traj_actions = []
    maps = []
    fromtos = []

    nrow, ncol = shape
    size = nrow * ncol

    print("Generating demos")

    bar = pyprind.ProgBar(n_maps)

    while len(traj_states) < n_maps:
        from_id = np.random.randint(size)
        to_id = np.random.randint(size)

        if from_id == to_id:
            continue

        map = new_map(shape=shape, n_obstacles=np.random.randint(low=1, high=max_n_obstacles))
        graph = to_sparse_adj_graph(map)

        sps, preds = scipy.sparse.csgraph.dijkstra(graph, directed=False, indices=[from_id], return_predecessors=True)

        from_x, from_y = from_id / ncol, from_id % ncol
        to_x, to_y = to_id / ncol, to_id % ncol

        if np.isinf(sps[0, to_id]):
            continue

        # Generate the states along the trajectory
        traj = [to_id]
        cur = to_id
        while cur != from_id:
            cur = preds[0, cur]
            traj.append(cur)
        traj = np.asarray(traj[::-1], dtype=np.int)

        flat_maps = np.tile(map.reshape((1, nrow, ncol)), (len(traj), 1, 1))
        flat_state = np.zeros((len(traj), nrow, ncol))
        flat_state[np.arange(len(traj)), traj / ncol, traj % ncol] = 1
        flat_goal = np.zeros((len(traj), nrow, ncol))
        flat_goal[:, to_x, to_y] = 1

        # import matplotlib.pyplot as plt; plt.set_cmap('gray'); plt.imshow(1 - (flat_maps[0] + flat_state.sum(0) *
        #                                                                        0.5), interpolation='nearest'); plt.show()

        flat_traj = np.stack([flat_maps, flat_state, flat_goal], axis=1)

        states, next_states = traj[:-1], traj[1:]
        actions = np.zeros_like(states) - 1

        # 0: left top
        # 1: top
        # 2: right top
        # 3: right
        # 4: right bottom
        # 5: bottom
        # 6: left bottom
        # 7: left
        actions[next_states == states - 1 - ncol] = 0  # left top
        actions[next_states == states - ncol] = 1  # top
        actions[next_states == states + 1 - ncol] = 2  # right top
        actions[next_states == states + 1] = 3  # right
        actions[next_states == states + 1 + ncol] = 4  # right bottom
        actions[next_states == states + ncol] = 5  # bottom
        actions[next_states == states - 1 + ncol] = 6  # left bottom
        actions[next_states == states - 1] = 7  # left

        assert np.all(actions >= 0)

        traj_states.append(flat_traj[:-1])
        traj_actions.append(actions)
        maps.append(map)
        fromtos.append((from_x, from_y, to_x, to_y))

        bar.update()

    if bar.active:
        bar.stop()

    return traj_states, traj_actions, maps, fromtos


# Now construct various NNs

class CNN(object):
    def __call__(self, shape):
        nrow, ncol = shape
        # debug = []
        input = L.InputLayer(shape=(None, 3, nrow, ncol))

        map_in = L.SliceLayer(input, indices=slice(0, 1), axis=1)
        state_in = L.SliceLayer(input, indices=slice(1, 2), axis=1)
        goal_in = L.SliceLayer(input, indices=slice(2, 3), axis=1)

        net_in = L.concat([map_in, goal_in], axis=1)

        net = L.Conv2DLayer(
            net_in, num_filters=50, filter_size=(3, 3), pad="same", nonlinearity=NL.rectify, convolution=wrapped_conv,
        )
        for idx in xrange(36):
            net = L.Conv2DLayer(
                net, num_filters=10, filter_size=(3, 3), pad="same", nonlinearity=NL.rectify, convolution=wrapped_conv,
            )

        net = OpLayer(
            net,
            extras=[state_in],
            op=lambda q_val, state_val: TT.sum(q_val * TT.addbroadcast(state_val, 1), axis=[2, 3]),
            shape_op=lambda q_shape, state_shape: (q_shape[0], q_shape[1]),
            name="q_out"
        )

        net = L.DenseLayer(net, num_units=8, nonlinearity=TT.nnet.softmax, name="output")
        return net


# class VIN(object):
#
#     def __init__(self, n_iter):
#         self.n_iter = n_iter
#
#     def __call__(self, shape):
#         nrow, ncol = shape
#         l_q = 10
#         n_iter = self.n_iter
#         input = L.InputLayer(shape=(None, nrow, ncol, 3), name="input")
#
#         bias = tf.get_variable(name="bias", shape=(150,), initializer=tf.random_normal_initializer(stddev=0.01))
#
#         w0 = tf.get_variable(name="w0", shape=(3, 3, 2, 150), initializer=tf.random_normal_initializer(stddev=0.01))
#         w1 = tf.get_variable(name="w1", shape=(3, 3, 150, 1), initializer=tf.random_normal_initializer(stddev=0.01))
#
#         w = tf.get_variable(name="w", shape=(3, 3, 1, l_q), initializer=tf.random_normal_initializer(stddev=0.01))
#         w_fb = tf.get_variable(name="w_fb", shape=(3, 3, 1, l_q), initializer=tf.random_normal_initializer(stddev=0.01))
#         w_bi = tf.concat(2, [w, w_fb])
#
#         map_in = L.SliceLayer(input, indices=slice(0, 1), axis=-1, name="map_in")
#         state_in = L.SliceLayer(input, indices=slice(1, 2), axis=-1, name="state_in")
#         goal_in = L.SliceLayer(input, indices=slice(2, 3), axis=-1, name="goal_in")
#
#         net_in = L.concat([map_in, goal_in], axis=3, name="net_in")
#
#         # initial conv layer over image+reward prior
#
#         h = L.Conv2DLayer(
#             net_in, num_filters=150, filter_size=(3, 3), pad="SAME", nonlinearity=None, name="h",
#             W=w0, b=bias,
#         )
#         r = L.Conv2DLayer(
#             h, num_filters=1, filter_size=(1, 1), pad="SAME", name="r0", nonlinearity=None,
#             W=w1, b=None
#         )
#
#         q = L.Conv2DLayer(
#             r, num_filters=l_q, filter_size=(3, 3), pad="SAME", name="q0", nonlinearity=None,
#             W=w, b=None
#         )
#         v = L.OpLayer(
#             q,
#             op=lambda x: tf.reduce_max(x, reduction_indices=-1, keep_dims=True),
#             shape_op=lambda shape: shape[:-1] + (1,),
#             name="v0"
#         )
#         for idx in xrange(1, n_iter + 1):
#             q = L.Conv2DLayer(
#                 L.concat([r, v], axis=3, name="rv%d" % idx),
#                 num_filters=l_q, filter_size=(3, 3), pad="SAME", name="q%d" % idx,
#                 W=w_bi, b=None, nonlinearity=None,
#             )
#             v = L.OpLayer(
#                 q,
#                 op=lambda x: tf.reduce_max(x, reduction_indices=-1, keep_dims=True),
#                 shape_op=lambda shape: shape[:-1] + (1,),
#                 name="v%d" % idx,
#             )
#         q = L.Conv2DLayer(
#             L.concat([r, v], axis=3, name="rv_last"),
#             num_filters=l_q, filter_size=(3, 3), pad="SAME", name="q_last",
#             W=w_bi, b=None, nonlinearity=None,
#         )
#
#         q_out = L.OpLayer(
#             q,
#             extras=[state_in],
#             op=lambda q_val, state_val: tf.reduce_sum(tf.reduce_sum(q_val * state_val, 1), 1),
#             shape_op=lambda q_shape, state_shape: (q_shape[0], q_shape[-1]),
#             name="q_out"
#         )
#
#         out = L.DenseLayer(
#             q_out,
#             num_units=8,
#             nonlinearity=tf.nn.softmax,
#             name="output"
#         )
#
#         # Do one last convolution
#         return out


def evaluate(policy, maps, fromtos, opt_trajlen, max_horizon):
    # Run the neural network on all the tasks at least once
    maps = np.asarray(maps)
    from_xs, from_ys, to_xs, to_ys = map(np.asarray, zip(*fromtos))
    goals = np.zeros_like(maps)
    goals[np.arange(len(maps)), to_xs, to_ys] = 1

    cur_xs, cur_ys = np.copy(from_xs), np.copy(from_ys)

    nrow, ncol = maps.shape[1:]

    n_trials = 0
    n_success = 0
    counter = np.zeros((len(maps),))
    tried = np.zeros((len(maps),))
    traj_difflen = []

    last_tried = 0
    progbar = pyprind.ProgBar(len(maps))

    for i in xrange(max_horizon):
        # print(i)
        counter += 1
        states = np.zeros_like(maps)
        states[np.arange(len(maps)), cur_xs, cur_ys] = 1
        stacked_obs = np.stack([maps, states, goals], axis=1)
        actions = policy(stacked_obs)
        steps = ACTION_MAP[actions]
        next_xs = cur_xs + steps[:, 0]
        next_ys = cur_ys + steps[:, 1]
        failed = logical_or(
            next_xs < 0,
            next_xs >= nrow,
            next_ys < 0,
            next_ys >= ncol,
            maps[np.arange(len(maps)), next_xs % nrow, next_ys % ncol]
        )
        success = np.logical_and(
            np.equal(next_xs, to_xs),
            np.equal(next_ys, to_ys),
        )
        done = np.logical_or(success, failed)
        tried[done] = 1
        n_trials += np.sum(done)
        n_success += np.sum(success)

        traj_difflen.extend(counter[success] - opt_trajlen[success])

        counter[done] = 0

        cur_xs = next_xs
        cur_ys = next_ys
        cur_xs[done] = from_xs[done]
        cur_ys[done] = from_ys[done]

        progbar.update(np.sum(tried) - last_tried)
        last_tried = np.sum(tried)

        if np.all(tried):
            if progbar.active:
                progbar.stop()
            break
    n_trials += np.sum(1 - tried)

    return n_success * 1.0 / n_trials, np.mean(traj_difflen)


def new_policy(state_var, action_prob_var):

    f_action_prob = ext.compile_function([state_var], action_prob_var, log_name="f_action_prob")

    def get_action(states):
        action_probs = f_action_prob(states)
        return special.weighted_sample_n(action_probs, np.arange(action_probs.shape[1]))

    return get_action


class MatlabData(object):
    def __init__(self, shape, data_fraction=1):
        if shape == (16, 16):
            file = "gridworld_16.mat"
        elif shape == (8, 8):
            file = "gridworld_8.mat"
        elif shape == (28, 28):
            file = "gridworld_28.mat"
        elif shape == (36, 36):
            file = "gridworld_36.mat"
        else:
            raise NotImplementedError

        file_path = os.path.join(config.PROJECT_PATH, 'sandbox/rocky/exp_data', file)

        matlab_data = sio.loadmat(file_path)

        im_data = matlab_data["batch_im_data"]
        im_data = 1 - im_data  # (im_data - 1)/255  # obstacles = 1, free zone = 0
        value_data = matlab_data["batch_value_data"]
        state1_data = matlab_data["state_x_data"]
        state2_data = matlab_data["state_y_data"]
        label_data = matlab_data["batch_label_data"]
        ydata = label_data.astype('int8')
        Xim_data = im_data.astype('float32')  # theano.config.floatX)
        Xim_data = Xim_data.reshape((-1,) + shape + (1,))
        Xval_data = value_data.astype('float32')  # theano.config.floatX)
        Xval_data = Xval_data.reshape((-1,) + shape + (1,))  # -1, 1, self.im_size[0], self.im_size[1])
        Xdata = np.append(Xim_data, Xval_data, axis=-1)
        S1data = state1_data.astype('int8')
        S2data = state2_data.astype('int8')

        all_training_samples = int(6 / 7.0 * Xdata.shape[0])
        training_samples = int(data_fraction * all_training_samples)

        self.Xtrain = Xdata[0:training_samples]
        self.S1train = S1data[0:training_samples]
        self.S2train = S2data[0:training_samples]
        self.ytrain = ydata[0:training_samples]

        self.Xtest = Xdata[all_training_samples:]
        self.S1test = S1data[all_training_samples:]
        self.S2test = S2data[all_training_samples:]
        self.ytest = ydata[all_training_samples:].flatten()


class GridworldBenchmark(object):
    def __init__(
            self,
            shape=(16, 16),
            max_n_obstacles=5,
            n_maps=10000,
            batch_size=128,
            n_actions=8,
            train_ratio=0.9,
            n_epochs=1000,
            eval_max_horizon=500,
            learning_rate=1e-3,
            lr_schedule=None,
            network=CNN(),
    ):
        self.shape = shape
        self.max_n_obstacles = max_n_obstacles
        self.n_maps = n_maps
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.train_ratio = train_ratio
        self.n_epochs = n_epochs
        self.eval_max_horizon = eval_max_horizon
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.network = network

    def train(self):
        # data = MatlabData(self.shape)

        # import ipdb; ipdb.set_trace()

        traj_states, traj_actions, maps, fromtos = gen_demos(
            shape=self.shape,
            max_n_obstacles=self.max_n_obstacles,
            n_maps=self.n_maps
        )

        n_train = int(np.ceil(self.train_ratio * self.n_maps))

        train_states = np.concatenate(traj_states[:n_train], axis=0)
        train_actions = np.concatenate(traj_actions[:n_train], axis=0)
        # convert actions to one-hot
        train_actions = np.eye(self.n_actions)[train_actions]
        train_maps = maps[:n_train]
        train_opt_trajlen = np.asarray(map(len, traj_states[:n_train]))
        train_fromtos = fromtos[:n_train]

        test_states = np.concatenate(traj_states[n_train:], axis=0)
        test_actions = np.concatenate(traj_actions[n_train:], axis=0)
        # convert actions to one-hot
        test_actions = np.eye(self.n_actions)[test_actions]
        test_maps = maps[n_train:]
        test_opt_trajlen = np.asarray(map(len, traj_states[n_train:]))
        test_fromtos = fromtos[n_train:]

        self.eval_max_horizon = max(np.max(train_opt_trajlen), np.max(test_opt_trajlen)) * 2

        dataset = BatchDataset(inputs=[train_states, train_actions], batch_size=self.batch_size)

        net = self.network(self.shape)

        state_var = TT.tensor4("state")
        action_var = TT.matrix("action")

        train_action_prob_var = L.get_output(net, state_var)
        test_action_prob_var = L.get_output(net, state_var, deterministic=True)

        train_loss_var = TT.mean(
            TT.sum(action_var * -TT.log(train_action_prob_var + 1e-8), -1)
        )
        train_err_var = TT.mean(
            TT.cast(TT.neq(TT.argmax(action_var, 1), TT.argmax(train_action_prob_var, 1)), floatX),
        )

        test_loss_var = TT.mean(
            TT.sum(action_var * -TT.log(test_action_prob_var + 1e-8), -1)
        )
        test_err_var = TT.mean(
            TT.cast(TT.neq(TT.argmax(action_var, 1), TT.argmax(test_action_prob_var, 1)), floatX),
        )

        params = L.get_all_params(net, trainable=True)
        # import ipdb; ipdb.set_trace()
        # params = filter(lambda x: isinstance(x, tf.Variable), params)

        lr_var = TT.scalar("lr")#tf.placeholder(dtype=tf.float32, shape=(), name="lr")

        updates = lasagne.updates.adam(train_loss_var, params, lr_var)

        f_train = ext.compile_function(
            inputs=[state_var, action_var, lr_var],
            outputs=[train_loss_var, train_err_var],
            updates=updates,
            log_name="f_train"
        )

        f_loss = ext.compile_function(
            inputs=[state_var, action_var, lr_var],
            outputs=[test_loss_var, test_err_var],
            log_name="f_loss",
        )

        # optimizer = tf.train.AdamOptimizer(learning_rate=lr_var, epsilon=1e-6)
        # train_op = optimizer.minimize(train_loss_var, var_list=params)

        if self.lr_schedule is not None:
            lr_list = []
            for lr, n_epochs in self.lr_schedule:
                lr_list.extend([lr] * n_epochs)
            assert len(lr_list) == self.n_epochs
            self.n_epochs = len(lr_list)
        else:
            lr_list = [self.learning_rate] * self.n_epochs

        # with tf.Session() as sess:
        #     sess.run(tf.initialize_all_variables())

            policy = new_policy(state_var, test_action_prob_var)

            for epoch in xrange(self.n_epochs):

                logger.log("Epoch %d" % epoch)
                bar = pyprind.ProgBar(len(train_states))

                train_losses = []
                train_errs = []

                if self.lr_schedule is not None and (epoch == 0 or lr_list[epoch] != lr_list[epoch - 1]):
                    raise NotImplementedError
                    # If learning rate changed, reset the optimizer state

                for batch_states, batch_actions in dataset.iterate():
                    # print(map(np.linalg.norm, sess.run(net.debug)))
                    # import ipdb; ipdb.set_trace()
                    train_loss, train_err = f_train(batch_states, batch_actions, lr_list[epoch])
                    bar.update(len(batch_states))
                    train_losses.append(train_loss)
                    train_errs.append(train_err)

                if bar.active:
                    bar.stop()

                logger.log("Evaluating error on test set")
                test_loss, test_err = f_loss(test_states, test_actions)

                logger.log("Evaluating policy")

                # subsample the same number of states from training data
                train_success_rate, avg_train_traj_len = evaluate(
                    policy, train_maps[:len(test_maps)], train_fromtos[:len(test_maps)], train_opt_trajlen[:len(
                        test_maps)], self.eval_max_horizon)
                test_success_rate, avg_test_traj_len = evaluate(
                    policy, test_maps, test_fromtos, test_opt_trajlen, self.eval_max_horizon)

                logger.record_tabular("Epoch", epoch)
                logger.record_tabular("AvgTrainLoss", np.mean(train_losses))
                logger.record_tabular("AvgTrainErr", np.mean(train_errs))
                logger.record_tabular("AvgTestLoss", test_loss)
                logger.record_tabular("AvgTestErr", test_err)
                logger.record_tabular("TrainSuccessRate", train_success_rate)
                logger.record_tabular("AvgTrainSuccessTrajLenDiff", avg_train_traj_len)
                logger.record_tabular("TestSuccessRate", test_success_rate)
                logger.record_tabular("AvgTestSuccessTrajLenDiff", avg_test_traj_len)
                logger.dump_tabular()
