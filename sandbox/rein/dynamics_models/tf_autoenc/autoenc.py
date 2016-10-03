import tensorflow as tf
import numpy as np
import math
from sandbox.rein.dynamics_models.utils import load_dataset_atari
from ops import lrelu, dense, conv2d, conv_transpose


class ConvAutoEncoder:
    """Convolutional/Deconvolutional autoencoder with shared weights.
    """

    def __init__(self,
                 input_shape=(42, 42, 1),
                 n_filters=(10, 10, 10),
                 filter_sizes=(3, 3, 3),
                 n_classes=10,
                 ):

        self._n_classes = n_classes

        # --
        self._x = tf.placeholder(tf.float32, (None,) + input_shape, name='x')
        current_input = self._x

        # --
        encoder = []
        shapes = []
        for layer_i, n_output in enumerate(n_filters):
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
            output = lrelu(
                tf.add(tf.nn.conv2d(
                    current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output

        # --
        # Flatten conv output.
        conv_out_shape = tf.shape(current_input)
        flattened_size = np.prod(current_input.get_shape().as_list()[1:])
        current_input = tf.reshape(current_input, (tf.pack([conv_out_shape[0], flattened_size])))
        W = tf.Variable(
            tf.random_uniform([flattened_size, 32],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([32]))
        self._z = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
        current_input = self._z
        W = tf.Variable(
            tf.random_uniform([32, 360],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([360]))
        current_input = lrelu(tf.matmul(current_input, W) + b)
        current_input = tf.reshape(current_input, tf.pack([tf.shape(current_input)[0], 6, 6, 10]))
        encoder.reverse()
        shapes.reverse()

        # --
        for layer_i, shape in enumerate(shapes):
            W = encoder[layer_i]
            b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
            output = lrelu(tf.add(
                tf.nn.conv2d_transpose(
                    current_input, W,
                    tf.pack([tf.shape(self._z)[0], shape[1], shape[2], shape[3]]),
                    strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output

        # --
        self._y = current_input
        self._cost = tf.reduce_sum(tf.square(self._y - self._x))

        # --
        learning_rate = 0.001
        self._optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self._cost)

    def transform(self, sess, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return sess.run(self.z, feed_dict={self.x: X})

    def generate(self, sess, z=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z is None:
            z = np.random.randint(0, 2, (10, 32))
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return sess.run(self.y, feed_dict={self.z: z})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X})

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

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def n_classes(self):
        return self._n_classes


def test_atari():
    import matplotlib.pyplot as plt

    atari_dataset = load_dataset_atari('/Users/rein/programming/datasets/dataset_42x42.pkl')
    atari_dataset['x'] = atari_dataset['x'].transpose((0, 2, 3, 1))
    ae = ConvAutoEncoder()

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    z = ae.transform(sess, atari_dataset['x'][0:10])
    print(z)
    y = ae.generate(sess, None)

    n_epochs = 2000
    for epoch_i in range(n_epochs):
        train = atari_dataset['x']
        sess.run(ae.optimizer, feed_dict={ae.x: train})
        print(epoch_i, sess.run(ae.cost, feed_dict={ae.x: train}))

    n_examples = 10
    recon = sess.run(ae.y, feed_dict={ae.x: atari_dataset['x'][0:n_examples]})
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
