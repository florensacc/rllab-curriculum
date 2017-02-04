from gpr_package.bin.tower_copter_policy import *
import tensorflow as tf
from copy import deepcopy as copy
import time

w_idx = 0


def weight_variable(shape):
    global w_idx
    W = tf.get_variable("W%d" % w_idx, shape=shape,
                        initializer=tf.contrib.layers.xavier_initializer())
    w_idx += 1
    return W


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def score(env, policy):
    rewards = []
    for idx in range(5):
        ob = env.reset()
        for t in range(env.horizon):
            action = policy(ob)
            ob, reward, _, _ = env.step(action)
            # if idx == 0:
            #     env.render()
        rewards.append(reward)
    print("\nFull evaluation")
    print("max(reward) = %f, mean(reward) = %f, std = %f\n" % (np.max(rewards), np.mean(rewards), np.std(rewards)))


# def preprocess(inp):
#     # XXXX This is a hack.
#     me = inp[:, 16:19]
#     dists = tf.concat(1, [inp[:, 19:22] - me, inp[:, 22:25] - me, inp[:, 25:28] - me, inp[:, 22:25] - inp[:, 25:28]])
#     dists = tf.log(dists * dists + 1e-4)
#     speed = tf.reduce_sum(inp[:, 8:16] * inp[:, 8:16], 1, keep_dims=True)
#     return tf.concat(1, [inp[:, :8], speed, me, dists])


def nn(inp, sizes):
    for idx, size1 in enumerate(sizes):
        if idx != 0:
            inp = tf.tanh(inp)
        size0 = inp.get_shape()[1].value
        W = weight_variable((size0, size1))
        b = bias_variable((size1,))
        inp = tf.matmul(inp, W) + b
    return inp


if __name__ == "__main__":

    task_id = get_task_from_text("ab")
    expr = Experiment(2, 600, bound=((-0.3, 0.3), (-0.3, 0.3), (0.0, 1.6)))
    env = expr.make(task_id)
    prescribed = CopterPolicy(task_id).get_action

    # score(env, prescribed)
    env.reset()

    obs_tf = tf.placeholder(tf.float32, [None, env.observation_space.flatten_dim])
    ys_tf = tf.placeholder(tf.float32, [None, env.action_space.flatten_dim])

    tmp = []
    loss = []
    y_pred = 0.0
    for _ in range(20):
        choice = np.random.randint(0, 4)
        layers = [64] * np.random.randint(1, 3)
        if choice == 0:
            # tmp.append(tf.tanh(nn(preprocess(obs_tf), layers + [env.action_space.flatten_dim])))
            tmp.append(tf.tanh(nn(obs_tf, layers + [env.action_space.flatten_dim])))
        elif choice == 1:
            # tmp.append(tf.tanh(nn(preprocess(obs_tf), layers + [env.action_space.flatten_dim])) * 0.2 + obs_tf[:, :8])
            tmp.append(tf.tanh(nn(obs_tf, layers + [env.action_space.flatten_dim])) * 0.2 + obs_tf[:, :8])
        elif choice == 2:
            tmp.append(tf.tanh(nn(obs_tf, layers + [env.action_space.flatten_dim])))
        elif choice == 3:
            tmp.append(tf.tanh(nn(obs_tf, layers + [env.action_space.flatten_dim])) * 0.2 + obs_tf[:, :8])
        else:
            assert (False)

        loss.append(tf.reduce_mean(tf.square(tmp[-1] - ys_tf)))
        tmp[-1] = tf.expand_dims(tmp[-1], 0)

    y_pred = tf.concat(0, tmp)

    loss = sum(loss) / len(loss)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    bs = 100
    xs = []
    idxs = []
    obs = [None] * bs

    print("Reseting")
    for seed in range(bs):
        print(seed)
        ob = env.reset_to(seed)
        steps = int(seed * env.horizon / bs)
        idxs.append(0)
        for _ in range(steps):
            y = copy(prescribed(ob))
            ob, _, _, _ = env.step(y)
            idxs[-1] += 1
        obs[seed], _ = copy(env._get_obs())
        xs.append(copy(env.x))

    idxs = np.array(idxs)
    print("Training")
    xs = np.stack(xs)

    loss_val = 10.0

    data = []

    last_display = time.time()
    for epoch in range(10000):
        obs_flatten = np.stack([env.observation_space.flatten(o) for o in obs])
        ys = []
        for i in range(bs):
            env.x = copy(xs[i])
            y = copy(prescribed(obs[i]))
            ob, _, _, _ = env.step(y)
            obs[i] = copy(ob)
            idxs[i] += 1
            if idxs[i] >= env.horizon:
                obs[i] = copy(env.reset())
                idxs[i] = 0
            xs[i] = copy(env.x)
            ys.append(y)

        # Later on the entire data comes from a neural network.
        data.append(copy((obs_flatten, ys)))

        if len(data) > 75:
            for tidx in range(20):
                # behavioural clonning.
                b = data[np.random.randint(0, len(data))]
                new_loss_val = sess.run([loss, train_op], {obs_tf: b[0], ys_tf: b[1]})[0]
                loss_val = 0.95 * loss_val + 0.05 * new_loss_val

        if epoch % 20 == 0:
            print("epoch = %d, loss = %f" % (epoch, loss_val))

        if time.time() - last_display > 120:
            last_display = time.time()


            def my_policy(o):
                tmp = sess.run(y_pred, {obs_tf: np.expand_dims(env.observation_space.flatten(o), 0)})
                tmp = np.squeeze(tmp)
                return np.median(tmp, 0)


            score(env, my_policy)
