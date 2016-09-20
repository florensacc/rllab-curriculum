import tensorflow as tf
import numpy as np
import math
from sandbox.rein.dynamics_models.utils import load_dataset_atari


class ConvAutoEncoder:
    """Convolutional/Deconvolutional autoencoder with shared weights.
    """

    def __init__(self,
                 input_shape=[None, 42, 42, 1],
                 n_filters=[1, 10, 10, 10],
                 filter_sizes=[3, 3, 3, 3],
                 ):

        # --
        self._x = tf.placeholder(
            tf.float32, input_shape, name='x')

        # --
        if len(self._x.get_shape()) == 2:
            x_dim = np.sqrt(self._x.get_shape().as_list()[1])
            if x_dim != int(x_dim):
                raise ValueError('Unsupported input dimensions')
            x_dim = int(x_dim)
            x_tensor = tf.reshape(
                self._x, [-1, x_dim, x_dim, n_filters[0]])
        elif len(self._x.get_shape()) == 4:
            x_tensor = self._x
        else:
            raise ValueError('Unsupported input dimensions')
        current_input = x_tensor

        # --
        encoder = []
        shapes = []
        for layer_i, n_output in enumerate(n_filters[1:]):
            n_input = current_input.get_shape().as_list()[3]
            shapes.append(current_input.get_shape().as_list())
            W = tf.Variable(
                tf.random_uniform([
                    filter_sizes[layer_i],
                    filter_sizes[layer_i],
                    n_input, n_output],
                    -1.0 / math.sqrt(n_input),
                    1.0 / math.sqrt(n_input)))
            b = tf.Variable(tf.zeros([n_output]))
            encoder.append(W)
            output = tf.nn.relu(
                tf.add(tf.nn.conv2d(
                    current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output

        # --
        self._z = current_input
        encoder.reverse()
        shapes.reverse()

        # --
        for layer_i, shape in enumerate(shapes):
            W = encoder[layer_i]
            b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
            output = tf.nn.relu(tf.add(
                tf.nn.conv2d_transpose(
                    current_input, W,
                    tf.pack([tf.shape(self._x)[0], shape[1], shape[2], shape[3]]),
                    strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output

        # --
        self._y = current_input
        self._cost = tf.reduce_sum(tf.square(self._y - x_tensor))

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def cost(self):
        return self._cost


def test_atari():
    import matplotlib.pyplot as plt

    atari_dataset = load_dataset_atari('/Users/rein/programming/datasets/dataset_42x42.pkl')
    atari_dataset['x'] = atari_dataset['x'].transpose((0, 2, 3, 1))
    ae = ConvAutoEncoder()

    # --
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae.cost)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    n_epochs = 10
    for epoch_i in range(n_epochs):
        train = atari_dataset['x']
        sess.run(optimizer, feed_dict={ae.x: train})
        print(epoch_i, sess.run(ae.cost, feed_dict={ae.x: train}))

    n_examples = 10
    recon = sess.run(ae.y, feed_dict={ae.x: atari_dataset['x'][0:n_examples]})
    print(recon.shape)
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(atari_dataset['x'][example_i], (42, 42)))
        axs[1][example_i].imshow(
            np.reshape(recon[example_i], (42, 42)))
    tf.train.SummaryWriter('/Users/rein/programming/tensorboard/logs', sess.graph)
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()


if __name__ == '__main__':
    test_atari()
