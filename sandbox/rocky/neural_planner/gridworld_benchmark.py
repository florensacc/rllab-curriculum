


import numpy as np
import random
import scipy.sparse
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
import pyprind
import os.path
import scipy.io as sio
from rllab.misc import special
from rllab.misc import logger
from rllab.optimizers.minibatch_dataset import BatchDataset
from rllab import config

ACTION_MAP = np.array([
    # Top
    [-1, 0],
    # Down
    [1, 0],
    # Right
    [0, 1],
    # Left
    [0, -1],
    # Top right
    [-1, 1],
    # Top left
    [-1, -1],
    # Bottom right
    [1, 1],
    # Bottom left
    [1, -1],
])

ACTION_LENGTH_MAP = np.array([
    1,
    1,
    1,
    1,
    np.sqrt(2),
    np.sqrt(2),
    np.sqrt(2),
    np.sqrt(2),
])

N_ACTIONS = 8


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

    for _ in range(n_obstacles):
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


def compute_optimal_trajlen(map, S1_list, S2_list):
    import ipdb;
    ipdb.set_trace()
    graph = to_sparse_adj_graph(map[:, :, 0])

    sps, preds = scipy.sparse.csgraph.dijkstra(map, directed=False, indices=[from_id], return_predecessors=True)


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

        flat_traj = np.stack([flat_maps, flat_state, flat_goal], axis=-1)

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

def resnet_batchnorm_nl(net, prev_net, nonlinearity):
    return L.NonlinearityLayer(
        L.ElemwiseSumLayer([L.batch_norm(net), prev_net]),
        nonlinearity=nonlinearity
    )


class CNN(object):

    def __init__(self, n_iter):
        self.n_iter = n_iter

    def build(self, shape):
        nrow, ncol = shape
        net_in = L.InputLayer(shape=(None, nrow, ncol, 2), name="input")

        net = L.batch_norm(net_in, name="bn0")

        net = L.Conv2DLayer(
            net, num_filters=20, filter_size=(3, 3), pad="SAME", nonlinearity=tf.nn.relu,
        )
        for idx in xrange(self.n_iter):
            net = resnet_batchnorm_nl(
                L.Conv2DLayer(
                    net, num_filters=20, filter_size=(3, 3), pad="SAME", nonlinearity=None,
                ), net, tf.nn.relu
            )

        net = L.Conv2DLayer(
            net, num_filters=N_ACTIONS, filter_size=(1, 1), pad="SAME", nonlinearity=None,
        )

        self.l_out = net
        self.l_out_variants = []
        self.params = L.get_all_params(self.l_out, trainable=True)


class VIN(object):
    def __init__(
            self, n_iter, n_q_filters, has_nonlinearity=False, untie_weights=False,
            batch_norm=False, has_bias=False):
        self.n_iter = n_iter
        self.n_q_filters = n_q_filters
        self.has_nonlinearity = has_nonlinearity
        self.untie_weights = untie_weights
        self.batch_norm = batch_norm
        self.has_bias = has_bias
        self.l_out_variants = []

    def build(self, shape):
        nrow, ncol = shape
        l_q = self.n_q_filters
        n_iter = self.n_iter
        net_in = L.InputLayer(shape=(None, nrow, ncol, 2))

        bias = tf.get_variable(name="bias", shape=(150,), initializer=tf.random_normal_initializer(stddev=0.01))

        w0 = tf.get_variable(name="w0", shape=(3, 3, 2, 150), initializer=tf.random_normal_initializer(stddev=0.01))
        w1 = tf.get_variable(name="w1", shape=(1, 1, 150, 1), initializer=tf.random_normal_initializer(stddev=0.01))

        w = tf.get_variable(name="w", shape=(3, 3, 1, l_q), initializer=tf.random_normal_initializer(stddev=0.01))
        w_fb = tf.get_variable(name="w_fb", shape=(3, 3, 1, l_q), initializer=tf.random_normal_initializer(stddev=0.01))
        w_bi = tf.concat(2, [w, w_fb])

        if self.has_nonlinearity:
            nonlinearity = tf.nn.relu
        else:
            nonlinearity = None

        # initial conv layer over image+reward prior

        h = L.Conv2DLayer(
            net_in, num_filters=150, filter_size=(3, 3), pad="SAME", nonlinearity=nonlinearity,
            W=w0, b=bias,
        )
        r = L.Conv2DLayer(
            h, num_filters=1, filter_size=(1, 1), pad="SAME", nonlinearity=nonlinearity,
            W=w1, b=None
        )

        q = L.Conv2DLayer(
            r, num_filters=l_q, filter_size=(3, 3), pad="SAME", nonlinearity=nonlinearity,
            W=w, b=None
        )
        v = L.OpLayer(
            q,
            op=lambda x: tf.reduce_max(x, reduction_indices=-1, keep_dims=True),
            shape_op=lambda shape: shape[:-1] + (1,),
        )

        params = [w0, bias, w1, w]
        if not self.untie_weights:
            params.append(w_fb)

        if self.untie_weights:
            q_W = tf.random_normal_initializer(stddev=0.01)
        else:
            q_W = w_bi

        for idx in xrange(n_iter):
            q = L.Conv2DLayer(
                L.concat([r, v], axis=3),
                num_filters=l_q, filter_size=(3, 3), pad="SAME",
                W=q_W, b=None, nonlinearity=nonlinearity,
            )
            v = L.OpLayer(
                q,
                op=lambda x: tf.reduce_max(x, reduction_indices=-1, keep_dims=True),
                shape_op=lambda shape: shape[:-1] + (1,),
            )
            if self.untie_weights:
                params.append(q.W)


        q = L.Conv2DLayer(
            L.concat([r, v], axis=3),
            num_filters=l_q, filter_size=(3, 3), pad="SAME",
            W=q_W, b=None, nonlinearity=nonlinearity,
        )
        if self.untie_weights:
            params.append(q.W)

        # The softmax operation is not yet applied!
        out = L.Conv2DLayer(
            q, num_filters=N_ACTIONS, filter_size=(1, 1), pad="SAME",
            nonlinearity=nonlinearity, b=None,
        )
        # At this moment the shape should be batch_size * height * width * N_actions
        # Should be more convenient to apply softmax outside

        self.l_out = out
        self.params = params + [out.W]  # [w0, bias, w1, w, w_fb, out.W]


class VIN1(object):
    def __init__(self, n_iter):
        self.n_iter = n_iter

    def build(self, shape):
        nrow, ncol = shape
        l_q = 10
        n_iter = self.n_iter
        net_in = L.InputLayer(shape=(None, nrow, ncol, 2))

        bias = tf.get_variable(name="bias", shape=(150,), initializer=tf.random_normal_initializer(stddev=0.01))

        w0 = tf.get_variable(name="w0", shape=(3, 3, 2, 150), initializer=tf.random_normal_initializer(stddev=0.01))
        w1 = tf.get_variable(name="w1", shape=(1, 1, 150, 1), initializer=tf.random_normal_initializer(stddev=0.01))

        w = tf.get_variable(name="w", shape=(3, 3, 1, l_q), initializer=tf.random_normal_initializer(stddev=0.01))
        w_fb = tf.get_variable(name="w_fb", shape=(3, 3, 1, l_q), initializer=tf.random_normal_initializer(stddev=0.01))
        w_bi = tf.concat(2, [w, w_fb])

        # initial conv layer over image+reward prior

        h = L.Conv2DLayer(
            net_in, num_filters=150, filter_size=(3, 3), pad="SAME", nonlinearity=tf.nn.relu,
            W=w0, b=bias,
        )
        r = L.Conv2DLayer(
            h, num_filters=1, filter_size=(1, 1), pad="SAME", nonlinearity=None,
            W=w1, b=None
        )

        q = L.Conv2DLayer(
            r, num_filters=l_q, filter_size=(3, 3), pad="SAME", nonlinearity=None,
            W=w, b=None
        )
        v = L.OpLayer(
            q,
            op=lambda x: tf.reduce_max(x, reduction_indices=-1, keep_dims=True),
            shape_op=lambda shape: shape[:-1] + (1,),
        )
        for idx in range(n_iter):
            q = L.Conv2DLayer(
                L.concat([r, v], axis=3),
                num_filters=l_q, filter_size=(3, 3), pad="SAME",
                W=w_bi, b=None, nonlinearity=None,
            )
            v = L.OpLayer(
                q,
                op=lambda x: tf.reduce_max(x, reduction_indices=-1, keep_dims=True),
                shape_op=lambda shape: shape[:-1] + (1,),
            )
        q = L.Conv2DLayer(
            L.concat([r, v], axis=3),
            num_filters=l_q, filter_size=(3, 3), pad="SAME",
            W=w_bi, b=None, nonlinearity=None,
        )

        # The softmax operation is not yet applied!
        out = L.Conv2DLayer(
            q, num_filters=N_ACTIONS, filter_size=(1, 1), pad="SAME",
            nonlinearity=None, b=None,
        )
        # At this moment the shape should be batch_size * height * width * N_actions
        # Should be more convenient to apply softmax outside

        self.l_out = out
        self.params = [w0, bias, w1, w, w_fb, out.W]


class VINMulti(object):
    def __init__(self, n_iter):
        self.n_iter = n_iter
        self.l_out = None
        self.params = None
        self.l_out_variants = None

    def build(self, shape):
        nrow, ncol = shape
        n_iter = self.n_iter
        l_q = 20
        net_in = L.InputLayer(shape=(None, nrow, ncol, 2))

        bias = tf.get_variable(name="bias", shape=(150,), initializer=tf.random_normal_initializer(stddev=0.01))

        w0 = tf.get_variable(name="w0", shape=(3, 3, 2, 150), initializer=tf.random_normal_initializer(stddev=0.01))
        w1 = tf.get_variable(name="w1", shape=(1, 1, 150, 1), initializer=tf.random_normal_initializer(stddev=0.01))

        w = tf.get_variable(name="w", shape=(3, 3, 1, l_q), initializer=tf.random_normal_initializer(stddev=0.01))
        w_fb = tf.get_variable(name="w_fb", shape=(3, 3, 1, l_q), initializer=tf.random_normal_initializer(stddev=0.01))
        w_bi = tf.concat(2, [w, w_fb])

        nonlinearity = None

        # initial conv layer over image+reward prior

        h = L.Conv2DLayer(
            net_in, num_filters=150, filter_size=(3, 3), pad="SAME", nonlinearity=nonlinearity,
            W=w0, b=bias,
        )
        r = L.Conv2DLayer(
            h, num_filters=1, filter_size=(1, 1), pad="SAME", nonlinearity=nonlinearity,
            W=w1, b=None
        )

        q = L.Conv2DLayer(
            r, num_filters=l_q, filter_size=(3, 3), pad="SAME", nonlinearity=nonlinearity,
            W=w, b=None
        )
        v = L.OpLayer(
            q,
            op=lambda x: tf.reduce_max(x, reduction_indices=-1, keep_dims=True),
            shape_op=lambda shape: shape[:-1] + (1,),
        )

        params = [w0, bias, w1, w, w_fb]

        for idx in xrange(n_iter):
            q = L.Conv2DLayer(
                L.concat([r, v], axis=3),
                num_filters=l_q, filter_size=(3, 3), pad="SAME",
                W=w_bi, b=None, nonlinearity=nonlinearity,
            )
            v = L.OpLayer(
                q,
                op=lambda x: tf.reduce_max(x, reduction_indices=-1, keep_dims=True),
                shape_op=lambda shape: shape[:-1] + (1,),
            )

        q = L.Conv2DLayer(
            L.concat([r, v], axis=3),
            num_filters=l_q, filter_size=(3, 3), pad="SAME",
            W=w_bi, b=None, nonlinearity=nonlinearity,
        )

        # The softmax operation is not yet applied!
        out = L.Conv2DLayer(
            q, num_filters=N_ACTIONS, filter_size=(1, 1), pad="SAME",
            nonlinearity=nonlinearity, b=None,
        )

        out_W = out.W
        # At this moment the shape should be batch_size * height * width * N_actions
        # Should be more convenient to apply softmax outside

        self.l_out = out
        self.params = params + [out_W]  # [w0, bias, w1, w, w_fb, out.W]

        self.l_out_variants = []

        for name, var_n_iter in [
            # ('3', 3),
            # ('x-10', n_iter - 10),
            # ('x-3', n_iter - 3),
            # ('x+3', n_iter + 3),
            # ('x+10', n_iter + 10)
        ]:
            q = L.Conv2DLayer(
                r, num_filters=l_q, filter_size=(3, 3), pad="SAME", nonlinearity=nonlinearity,
                W=w, b=None
            )
            v = L.OpLayer(
                q,
                op=lambda x: tf.reduce_max(x, reduction_indices=-1, keep_dims=True),
                shape_op=lambda shape: shape[:-1] + (1,),
            )

            for idx in xrange(var_n_iter):
                q = L.Conv2DLayer(
                    L.concat([r, v], axis=3),
                    num_filters=l_q, filter_size=(3, 3), pad="SAME",
                    W=w_bi, b=None, nonlinearity=nonlinearity,
                )
                v = L.OpLayer(
                    q,
                    op=lambda x: tf.reduce_max(x, reduction_indices=-1, keep_dims=True),
                    shape_op=lambda shape: shape[:-1] + (1,),
                )

            q = L.Conv2DLayer(
                L.concat([r, v], axis=3),
                num_filters=l_q, filter_size=(3, 3), pad="SAME",
                W=w_bi, b=None, nonlinearity=nonlinearity,
            )

            # The softmax operation is not yet applied!
            out = L.Conv2DLayer(
                q, num_filters=N_ACTIONS, filter_size=(1, 1), pad="SAME", nonlinearity=nonlinearity,
                W=out_W, b=None,
            )

            self.l_out_variants.append((name, out))


def evaluate(policy, maps, fromtos, opt_trajlen, max_horizon):
    # Run the neural network on all the tasks at least once
    maps = np.asarray(maps)
    from_xs, from_ys, to_xs, to_ys = list(map(np.asarray, list(zip(*fromtos))))
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

    for i in range(max_horizon):
        counter += 1
        states = np.zeros_like(maps)
        states[np.arange(len(maps)), cur_xs, cur_ys] = 1
        stacked_obs = np.stack([maps, states, goals], axis=-1)
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


def new_policy(sess, state_var, action_prob_var):
    def get_action(states):
        action_probs = sess.run(action_prob_var, feed_dict={state_var: states})
        return special.weighted_sample_n(action_probs, np.arange(action_probs.shape[1]))

    return get_action


def gather_nd(params, indices, name=None):
    shape = params.get_shape().as_list()
    rank = len(shape)
    flat_params = tf.reshape(params, [-1])
    multipliers = [reduce(lambda x, y: x * y, shape[i + 1:], 1) for i in range(0, rank)]
    indices_unpacked = tf.unpack(tf.transpose(indices, [rank - 1] + range(0, rank - 1), name))
    flat_indices = sum([a * b for a, b in zip(multipliers, indices_unpacked)])
    return tf.gather(flat_params, flat_indices, name=name)


def evaluate_matlab(Xtest, S1test, S2test, opt_traj_lens, state_var, slice_1d_ids_var, test_action_prob_var):
    state_batch_size = S1test.shape[-1]

    from_xs = S1test.flatten()
    from_ys = S2test.flatten()

    _, to_xs, to_ys = np.where(Xtest[:, :, :, 1])

    to_xs = np.repeat(to_xs, state_batch_size)
    to_ys = np.repeat(to_ys, state_batch_size)

    cur_xs = np.copy(from_xs)
    cur_ys = np.copy(from_ys)

    nrow, ncol = Xtest.shape[1], Xtest.shape[2]

    n_trials = 0
    n_success = 0

    tried = np.zeros_like(cur_xs)
    traj_difflen = []

    last_tried = 0

    test_traj_lens = np.zeros_like(cur_xs, dtype=np.float32)

    progbar = pyprind.ProgBar(cur_xs.size)

    max_horizon = max(int(np.max(opt_traj_lens)) + 10, (nrow + ncol) * 2)

    for i in xrange(max_horizon):

        test_1st_ids = np.repeat(np.arange(len(Xtest)), state_batch_size)
        test_all_ids = np.stack([
            test_1st_ids, cur_xs, cur_ys,
        ], axis=1)
        flat_test_ids = test_all_ids[:, 0] * nrow * ncol + \
                        test_all_ids[:, 1] * ncol + \
                        test_all_ids[:, 2]
        test_action_probs = tf.get_default_session().run(
            test_action_prob_var,
            feed_dict={
                state_var: Xtest,
                slice_1d_ids_var: flat_test_ids,
            }
        )
        actions = special.weighted_sample_n(test_action_probs, np.arange(N_ACTIONS))

        steps = ACTION_MAP[actions]
        next_xs = cur_xs + steps[:, 0]
        next_ys = cur_ys + steps[:, 1]
        test_traj_lens += ACTION_LENGTH_MAP[actions]
        failed = logical_or(
            next_xs < 0,
            next_xs >= nrow,
            next_ys < 0,
            next_ys >= ncol,
            Xtest[test_1st_ids, next_xs % nrow, next_ys % ncol, 0]
        )
        success = np.logical_and(
            np.equal(next_xs, to_xs),
            np.equal(next_ys, to_ys),
        )
        done = np.logical_or(success, failed)
        tried[done] = 1
        n_trials += np.sum(done)
        n_success += np.sum(success)

        traj_difflen.extend(test_traj_lens[success] - opt_traj_lens[success])

        test_traj_lens[done] = 0

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
    if progbar.active:
        progbar.stop()

    return n_success * 1.0 / n_trials, traj_difflen


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
        elif shape == (17, 17):
            file = "gridworld_17.mat"
        elif shape == (9, 9):
            file = "gridworld_9.mat"
        elif shape == (29, 29):
            file = "gridworld_29.mat"
        else:
            raise NotImplementedError

        file_path = os.path.join(config.PROJECT_PATH, 'sandbox/rocky/exp_data', file)

        matlab_data = sio.loadmat(file_path)

        im_data = matlab_data["batch_im_data"]
        im_data = (im_data - 1) / 255
        # Note: the value at the goal is 10 instead of 1. Worth investigating whether this matters
        value_data = matlab_data["batch_value_data"]
        state1_data = matlab_data["state_x_data"]
        state2_data = matlab_data["state_y_data"]
        label_data = matlab_data["batch_label_data"]
        ydata = label_data.astype('int')
        Xim_data = im_data.astype('float32')
        Xim_data = Xim_data.reshape((-1,) + shape + (1,))
        Xval_data = value_data.astype('float32')
        Xval_data = Xval_data.reshape((-1,) + shape + (1,))
        Xdata = np.append(Xim_data, Xval_data, axis=-1)
        S1data = state1_data.astype('int')
        S2data = state2_data.astype('int')

        all_training_samples = int(6 / 7.0 * Xdata.shape[0])
        training_samples = int(data_fraction * all_training_samples)

        self.Xtrain = Xdata[0:training_samples]  # N * height * width * 2
        self.S1train = S1data[0:training_samples]  # N * state_batch_size
        self.S2train = S2data[0:training_samples]  # N * state_batch_size
        self.ytrain = ydata[0:training_samples]  # N * state_batch_size

        self.Xtest = Xdata[all_training_samples:]
        self.S1test = S1data[all_training_samples:]
        self.S2test = S2data[all_training_samples:]
        self.ytest = ydata[all_training_samples:]  # .flatten()

        self.shape = shape

    def train(self, benchmark):

        state_batch_size = self.S1train.shape[1]

        state_var = tf.placeholder(dtype=tf.float32, shape=(None,) + self.shape + (2,), name="state")

        opt_traj_lens = []

        # Figure out the optimal trajectory lengths for the portion of data we'd like to test
        logger.log("computing optimal trajectory lengths for test data...")
        bar = pyprind.ProgBar(len(self.Xtest))
        for idx in xrange(len(self.Xtest)):
            map_ = self.Xtest[idx, :, :, 0]
            graph = to_sparse_adj_graph(map_)
            to_x, to_y = map(int, np.where(self.Xtest[idx, :, :, 1]))
            to_id = to_x * self.shape[1] + to_y
            from_ids = self.S1test[idx] * self.shape[1] + self.S2test[idx]
            sps, preds = scipy.sparse.csgraph.dijkstra(graph, directed=False, indices=[to_id], return_predecessors=True)
            opt_traj_lens.append(sps[:, from_ids].flatten())
            bar.update(1)
        if bar.active:
            bar.stop()

        opt_traj_lens = np.asarray(opt_traj_lens).flatten()

        # instead of passing in separate coordinates, pass in already well-formed 1D ids
        slice_1d_ids = tf.placeholder(dtype=tf.int32, shape=(None,), name="slice_id_ids")
        action_var = tf.placeholder(dtype=tf.float32, shape=(None, N_ACTIONS), name="action")

        net = benchmark.network  # (self.shape)
        net.build(self.shape)

        train_action_logits_var = L.get_output(net.l_out, state_var, phase='train')
        test_action_logits_var = L.get_output(net.l_out, state_var, phase='test')
        variant_test_action_logits_vars = []
        for _, var_out in net.l_out_variants:
            variant_test_action_logits_vars.append(L.get_output(var_out, state_var, phase='test'))

        # batch_size = tf.shape(state_var)[0]

        # batch_ids = tf.reshape(
        #     tf.tile(
        #         tf.reshape(tf.range(batch_size), (1, -1)),
        #         (state_batch_size, 1)
        #     ), (-1,)
        # )

        train_action_prob_var = tf.nn.softmax(
            tf.gather(tf.reshape(train_action_logits_var, (-1, N_ACTIONS)), slice_1d_ids),
        )
        test_action_prob_var = tf.nn.softmax(
            tf.gather(tf.reshape(test_action_logits_var, (-1, N_ACTIONS)), slice_1d_ids),
        )
        variant_test_action_prob_vars = []
        for variant_test_action_logits_var in variant_test_action_logits_vars:
            variant_test_action_prob_vars.append(
                tf.nn.softmax(
                    tf.gather(tf.reshape(variant_test_action_logits_var, (-1, N_ACTIONS)), slice_1d_ids),
                )
            )

        train_loss_var = tf.reduce_mean(
            tf.reduce_sum(action_var * -tf.log(train_action_prob_var + 1e-8), -1)
        )
        train_err_var = tf.reduce_mean(
            tf.cast(tf.not_equal(tf.arg_max(action_var, 1), tf.arg_max(train_action_prob_var, 1)), tf.float32)
        )

        test_loss_var = tf.reduce_mean(
            tf.reduce_sum(action_var * -tf.log(test_action_prob_var + 1e-8), -1)
        )
        test_err_var = tf.reduce_mean(
            tf.cast(tf.not_equal(tf.arg_max(action_var, 1), tf.arg_max(test_action_prob_var, 1)), tf.float32)
        )

        variant_test_loss_vars = []
        variant_test_err_vars = []
        for variant_test_action_prob_var in variant_test_action_prob_vars:
            variant_test_loss_vars.append(
                tf.reduce_mean(
                    tf.reduce_sum(action_var * -tf.log(variant_test_action_prob_var + 1e-8), -1)
                )
            )
            variant_test_err_vars.append(
                tf.reduce_mean(
                    tf.cast(tf.not_equal(tf.arg_max(action_var, 1), tf.arg_max(variant_test_action_prob_var, 1)),
                            tf.float32)
                )
            )

        params = net.params  # L.get_all_params(net, trainable=True)
        # params = filter(lambda x: isinstance(x, tf.Variable), params)

        # import ipdb; ipdb.set_trace()

        lr_var = tf.placeholder(dtype=tf.float32, shape=(), name="lr")

        # optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_var, epsilon=1e-6)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr_var)  # , epsilon=1e-6)
        gradients = tf.gradients(train_loss_var, xs=params)
        noised_gradients = [
            g + tf.random_normal(g.get_shape(), stddev=benchmark.gradient_noise_scale) for g in gradients
            ]
        train_op = optimizer.apply_gradients(zip(noised_gradients, params))
        # train_op = optimizer.minimize(train_loss_var, var_list=params)

        # if benchmark.lr_schedule is not None:
        #     lr_list = []
        #     for lr, n_epochs in benchmark.lr_schedule:
        #         lr_list.extend([lr] * n_epochs)
        #     assert len(lr_list) == benchmark.n_epochs
        #     benchmark.n_epochs = len(lr_list)
        # else:
        #     lr_list = [benchmark.learning_rate] * benchmark.n_epochs

        dataset = BatchDataset([self.Xtrain, self.S1train, self.S2train, self.ytrain], batch_size=benchmark.batch_size)

        # prev_loss = np.inf
        best_loss = np.inf
        best_params = None
        n_no_improvement = 0

        learning_rate = benchmark.learning_rate

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            # policy = new_policy(sess, state_var, test_action_prob_var)

            for epoch in range(benchmark.n_epochs):

                logger.log("Epoch %d" % epoch)
                bar = pyprind.ProgBar(len(self.Xtrain))

                train_losses = []
                train_errs = []

                for batch in dataset.iterate():
                    batch_states, batch_s1, batch_s2, batch_ys = batch
                    batch_1st_ids = np.repeat(np.arange(len(batch_states)), state_batch_size)
                    batch_all_ids = np.stack([
                        batch_1st_ids, batch_s1.flatten(), batch_s2.flatten()
                    ], axis=1)
                    flat_ids = batch_all_ids[:, 0] * self.shape[0] * self.shape[1] + \
                               batch_all_ids[:, 1] * self.shape[1] + \
                               batch_all_ids[:, 2]
                    flat_ys = np.eye(N_ACTIONS)[batch_ys.flatten()]

                    train_loss, train_err, _ = sess.run(
                        [train_loss_var, train_err_var, train_op],
                        feed_dict={
                            state_var: batch_states,
                            slice_1d_ids: flat_ids,
                            action_var: flat_ys,
                            lr_var: learning_rate,
                        }
                    )

                    bar.update(len(batch_states))
                    train_losses.append(train_loss)
                    train_errs.append(train_err)

                if bar.active:
                    bar.stop()

                avg_train_loss = np.mean(train_losses)
                if avg_train_loss > best_loss:
                    n_no_improvement += 1
                else:
                    n_no_improvement = 0
                    best_loss = avg_train_loss
                    # collect best params
                    best_params = sess.run(params)

                test_1st_ids = np.repeat(np.arange(len(self.Xtest)), state_batch_size)
                test_all_ids = np.stack([
                    test_1st_ids, self.S1test.flatten(), self.S2test.flatten()
                ], axis=1)
                flat_test_ids = test_all_ids[:, 0] * self.shape[0] * self.shape[1] + \
                                test_all_ids[:, 1] * self.shape[1] + \
                                test_all_ids[:, 2]
                flat_test_ys = np.eye(N_ACTIONS)[self.ytest.flatten()]

                logger.log("Evaluating error on test set")
                test_loss, test_err = sess.run(
                    [test_loss_var, test_err_var],
                    feed_dict={
                        state_var: self.Xtest,
                        slice_1d_ids: flat_test_ids,
                        action_var: flat_test_ys,
                    }
                )
                #
                logger.log("Evaluating policy")

                test_success_rate, test_traj_lens = evaluate_matlab(
                    self.Xtest, self.S1test, self.S2test, opt_traj_lens, state_var, slice_1d_ids, test_action_prob_var
                )

                logger.record_tabular("Epoch", epoch)
                logger.record_tabular("LearningRate", learning_rate)
                logger.record_tabular("NoImprovementEpochs", n_no_improvement)
                logger.record_tabular("AvgTrainLoss", np.mean(train_losses))
                logger.record_tabular("AvgTrainErr", np.mean(train_errs))
                logger.record_tabular("AvgTestLoss", test_loss)
                logger.record_tabular("AvgTestErr", test_err)
                logger.record_tabular("TestSuccessRate", test_success_rate)
                logger.record_tabular("AvgTestSuccessTrajLenDiff", np.mean(test_traj_lens))
                logger.record_tabular("MaxTestSuccessTrajLenDiff", np.max(test_traj_lens))
                logger.record_tabular("MinTestSuccessTrajLenDiff", np.min(test_traj_lens))

                for (variant_name, _), variant_test_action_prob_var in zip(
                        net.l_out_variants, variant_test_action_prob_vars):
                    variant_test_success_rate, variant_test_traj_lens = evaluate_matlab(
                        self.Xtest, self.S1test, self.S2test, opt_traj_lens, state_var, slice_1d_ids,
                        variant_test_action_prob_var
                    )
                    logger.record_tabular("TestSuccessRate(%s)" % variant_name, variant_test_success_rate)
                    logger.record_tabular("AvgTestSuccessTrajLenDiff(%s)" % variant_name,
                                          np.mean(variant_test_traj_lens))
                    logger.record_tabular("MaxTestSuccessTrajLenDiff(%s)" % variant_name,
                                          np.max(variant_test_traj_lens))
                    logger.record_tabular("MinTestSuccessTrajLenDiff(%s)" % variant_name,
                                          np.min(variant_test_traj_lens))

                logger.dump_tabular()

                if n_no_improvement >= benchmark.no_improvement_tolerance:
                    learning_rate *= 0.5
                    logger.log("No improvement for %d epochs. Reducing learning rate to %f" % (n_no_improvement,
                                                                                               learning_rate))
                    n_no_improvement = 0
                    # restore to best params
                    sess.run([tf.assign(p, pv) for p, pv in zip(params, best_params)])


class GridworldBenchmark(object):
    def __init__(
            self,
            network,
            shape=(16, 16),
            max_n_obstacles=5,
            n_maps=10000,
            batch_size=128,
            n_actions=8,
            train_ratio=0.9,
            n_epochs=1000,
            eval_max_horizon=500,
            learning_rate=1e-3,
            gradient_noise_scale=0.,
            no_improvement_tolerance=5,
            # lr_schedule=None,
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
        self.no_improvement_tolerance = no_improvement_tolerance
        # self.lr_schedule = lr_schedule
        self.network = network
        self.gradient_noise_scale = gradient_noise_scale

    def train(self):
        data = MatlabData(self.shape)
        data.train(self)
        return

        # traj_states, traj_actions, maps, fromtos = gen_demos(
        #     shape=self.shape,
        #     max_n_obstacles=self.max_n_obstacles,
        #     n_maps=self.n_maps
        # )
        #
        # n_train = int(np.ceil(self.train_ratio * self.n_maps))
        #
        # train_states = np.concatenate(traj_states[:n_train], axis=0)
        # train_actions = np.concatenate(traj_actions[:n_train], axis=0)
        # # convert actions to one-hot
        # train_actions = np.eye(self.n_actions)[train_actions]
        # train_maps = maps[:n_train]
        # train_opt_trajlen = np.asarray(map(len, traj_states[:n_train]))
        # train_fromtos = fromtos[:n_train]
        #
        # test_states = np.concatenate(traj_states[n_train:], axis=0)
        # test_actions = np.concatenate(traj_actions[n_train:], axis=0)
        # # convert actions to one-hot
        # test_actions = np.eye(self.n_actions)[test_actions]
        # test_maps = maps[n_train:]
        # test_opt_trajlen = np.asarray(map(len, traj_states[n_train:]))
        # test_fromtos = fromtos[n_train:]
        #
        # self.eval_max_horizon = max(np.max(train_opt_trajlen), np.max(test_opt_trajlen)) * 2
        #
        # dataset = BatchDataset(inputs=[train_states, train_actions], batch_size=self.batch_size)
        #
        # net = self.network(self.shape)
        #
        # state_var = tf.placeholder(dtype=tf.float32, shape=(None,) + self.shape + (3,), name="state")
        # action_var = tf.placeholder(dtype=tf.float32, shape=(None, self.n_actions), name="action")
        #
        # train_action_prob_var = L.get_output(net, state_var, phase='train')
        # test_action_prob_var = L.get_output(net, state_var, phase='test')
        #
        # train_loss_var = tf.reduce_mean(
        #     tf.reduce_sum(action_var * -tf.log(train_action_prob_var + 1e-8), -1)
        # )
        # train_err_var = tf.reduce_mean(
        #     tf.cast(tf.not_equal(tf.arg_max(action_var, 1), tf.arg_max(train_action_prob_var, 1)), tf.float32)
        # )
        #
        # test_loss_var = tf.reduce_mean(
        #     tf.reduce_sum(action_var * -tf.log(test_action_prob_var + 1e-8), -1)
        # )
        # test_err_var = tf.reduce_mean(
        #     tf.cast(tf.not_equal(tf.arg_max(action_var, 1), tf.arg_max(test_action_prob_var, 1)), tf.float32)
        # )
        #
        # params = L.get_all_params(net, trainable=True)
        # params = filter(lambda x: isinstance(x, tf.Variable), params)
        #
        # lr_var = tf.placeholder(dtype=tf.float32, shape=(), name="lr")
        #
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_var, epsilon=1e-6)
        # train_op = optimizer.minimize(train_loss_var, var_list=params)
        #
        # if self.lr_schedule is not None:
        #     lr_list = []
        #     for lr, n_epochs in self.lr_schedule:
        #         lr_list.extend([lr] * n_epochs)
        #     assert len(lr_list) == self.n_epochs
        #     self.n_epochs = len(lr_list)
        # else:
        #     lr_list = [self.learning_rate] * self.n_epochs
        #
        # with tf.Session() as sess:
        #     sess.run(tf.initialize_all_variables())
        #
        #     policy = new_policy(sess, state_var, test_action_prob_var)
        #
        #     for epoch in xrange(self.n_epochs):
        #
        #         logger.log("Epoch %d" % epoch)
        #         bar = pyprind.ProgBar(len(train_states))
        #
        #         train_losses = []
        #         train_errs = []
        #
        #         # if epoch == 0 or lr_list[epoch] != lr_list[epoch - 1]:
        #         #     # If learning rate changed, reset the optimizer state
        #         #     logger.log("Resetting optimizer state..")
        #         #     if isinstance(optimizer, tf.train.AdamOptimizer):
        #         #         vars = optimizer._slots['m'].values() + optimizer._slots['v'].values()
        #         #         var_vals = sess.run(vars)
        #         #         ops = []
        #         #         for var, val in zip(vars, var_vals):
        #         #             ops.append(tf.assign(var, np.zeros_like(val)))
        #         #         sess.run(ops)
        #         #     elif isinstance(optimizer, tf.train.RMSPropOptimizer):
        #         #         vars = optimizer._slots['rms'].values() + optimizer._slots['momentum'].values()
        #         #         var_vals = sess.run(vars)
        #         #         ops = []
        #         #         for var, val in zip(vars, var_vals):
        #         #             ops.append(tf.assign(var, np.zeros_like(val)))
        #         #         sess.run(ops)
        #         #     else:
        #         #         import ipdb; ipdb.set_trace()
        #
        #         for batch_states, batch_actions in dataset.iterate():
        #             # print(map(np.linalg.norm, sess.run(net.debug)))
        #             # import ipdb; ipdb.set_trace()
        #             train_loss, train_err, _ = sess.run(
        #                 [train_loss_var, train_err_var, train_op],
        #                 feed_dict={
        #                     state_var: batch_states,
        #                     action_var: batch_actions,
        #                     lr_var: lr_list[epoch],
        #                 }
        #             )
        #             bar.update(len(batch_states))
        #             train_losses.append(train_loss)
        #             train_errs.append(train_err)
        #
        #         if bar.active:
        #             bar.stop()
        #
        #         logger.log("Evaluating error on test set")
        #         test_loss, test_err = sess.run(
        #             [test_loss_var, test_err_var],
        #             feed_dict={
        #                 state_var: test_states,
        #                 action_var: test_actions
        #             }
        #         )
        #
        #         logger.log("Evaluating policy")
        #
        #         # subsample the same number of states from training data
        #         train_success_rate, avg_train_traj_len = evaluate(
        #             policy, train_maps[:len(test_maps)], train_fromtos[:len(test_maps)], train_opt_trajlen[:len(
        #                 test_maps)], self.eval_max_horizon)
        #         test_success_rate, avg_test_traj_len = evaluate(
        #             policy, test_maps, test_fromtos, test_opt_trajlen, self.eval_max_horizon)
        #
        #         logger.record_tabular("Epoch", epoch)
        #         logger.record_tabular("AvgTrainLoss", np.mean(train_losses))
        #         logger.record_tabular("AvgTrainErr", np.mean(train_errs))
        #         logger.record_tabular("AvgTestLoss", test_loss)
        #         logger.record_tabular("AvgTestErr", test_err)
        #         logger.record_tabular("TrainSuccessRate", train_success_rate)
        #         logger.record_tabular("AvgTrainSuccessTrajLenDiff", avg_train_traj_len)
        #         logger.record_tabular("TestSuccessRate", test_success_rate)
        #         logger.record_tabular("AvgTestSuccessTrajLenDiff", avg_test_traj_len)
        #         logger.dump_tabular()
