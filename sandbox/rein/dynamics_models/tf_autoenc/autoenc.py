import tensorflow as tf
import numpy as np
from sandbox.rein.dynamics_models.utils import load_dataset_atari
import sandbox.rocky.tf.core.layers as L


class BinaryCodeConvAE:
    """Convolutional/Deconvolutional autoencoder with shared weights.
    """

    def __init__(self,
                 input_shape=(42, 42, 1),
                 ):
        self._x = tf.placeholder(tf.float32, shape=(None,) + input_shape, name="input")
        l_in = L.InputLayer(shape=(None,) + input_shape, input_var=self._x, name="input_layer")
        l_conv_1 = L.Conv2DLayer(
            l_in,
            num_filters=96,
            filter_size=5,
            stride=(2, 2),
            pad='VALID',
            nonlinearity=tf.nn.relu,
            name='conv_1',
            weight_normalization=True,
        )
        l_conv_1_bn = L.batch_norm(l_conv_1)
        l_conv_2 = L.Conv2DLayer(
            l_conv_1_bn,
            num_filters=96,
            filter_size=5,
            stride=(2, 2),
            pad='VALID',
            nonlinearity=tf.nn.relu,
            name='conv_2',
            weight_normalization=True,
        )
        l_conv_2_bn = L.batch_norm(l_conv_2)
        l_flatten_1 = L.FlattenLayer(l_conv_2_bn)
        l_dense_1 = L.DenseLayer(
            l_flatten_1,
            num_units=128,
            nonlinearity=tf.nn.relu,
            name='hidden_1',
            W=L.XavierUniformInitializer(),
            b=tf.zeros_initializer,
            weight_normalization=True
        )
        l_dense_1_bn = L.batch_norm(l_dense_1)
        l_dense_2 = L.DenseLayer(
            l_dense_1_bn,
            num_units=32,
            nonlinearity=tf.nn.sigmoid,
            name='binary_code',
            W=L.XavierUniformInitializer(),
            b=tf.zeros_initializer,
            weight_normalization=True
        )
        l_dense_2_bn = L.batch_norm(l_dense_2)
        self._z = l_dense_2_bn
        l_dense_3 = L.DenseLayer(
            l_dense_2_bn,
            num_units=np.prod(l_conv_2.output_shape[1:]),
            nonlinearity=tf.nn.relu,
            name='hidden_2',
            W=L.XavierUniformInitializer(),
            b=tf.zeros_initializer,
            weight_normalization=True
        )
        l_dense_3_bn = L.batch_norm(l_dense_3)
        l_reshp_1 = L.ReshapeLayer(
            l_dense_3_bn,
            (-1,) + l_conv_2_bn.output_shape[1:]
        )
        l_deconv_1 = L.TransposedConv2DLayer(
            l_reshp_1,
            num_filters=96,
            filter_size=5,
            stride=(2, 2),
            crop='VALID',
            nonlinearity=tf.nn.relu,
            name='deconv_1',
            weight_normalization=True,
        )
        l_deconv_1_bn = L.batch_norm(l_deconv_1)
        l_deconv_2 = L.TransposedConv2DLayer(
            l_deconv_1_bn,
            num_filters=1,
            filter_size=6,
            stride=(2, 2),
            crop='VALID',
            nonlinearity=tf.nn.sigmoid,
            name='deconv_2',
            weight_normalization=True,
        )
        l_out = l_deconv_2
        # l_deconv_2_bn = L.batch_norm(l_deconv_2)
        # l_out = L.Conv2DLayer(l_deconv_2_bn, 1, 1, pad='SAME', nonlinearity=tf.nn.sigmoid)
        self._y = L.get_output(l_out)
        self._z_in = tf.placeholder(tf.float32, shape=(None, 32), name="input")
        self._y_gen = L.get_output(l_out, {l_dense_2_bn: self._z_in}, deterministic=True)

        print(l_conv_1.output_shape)
        print(l_conv_2.output_shape)
        print(l_deconv_1.output_shape)
        print(l_deconv_2.output_shape)
        print(l_out.output_shape)

        # --
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
            z = np.random.randint(0, 2, (1, 32))
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return sess.run(self._y_gen, feed_dict={self._z_in: z})

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
    ae = BinaryCodeConvAE()

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    n_epochs = 1000
    for epoch_i in range(n_epochs):
        train = atari_dataset['x']
        sess.run(ae.optimizer, feed_dict={ae.x: train})
        print(epoch_i, sess.run(ae.cost, feed_dict={ae.x: train}))

    n_examples = 10
    recon = sess.run(ae.y, feed_dict={ae.x: atari_dataset['x'][0:n_examples]})
    fig, axs = plt.subplots(2, n_examples, figsize=(20, 4))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(atari_dataset['x'][example_i], (42, 42)),
            cmap='Greys_r', vmin=0, vmax=1, interpolation='none')
        axs[0][example_i].xaxis.set_visible(False)
        axs[0][example_i].yaxis.set_visible(False)
        axs[1][example_i].imshow(
            np.reshape(recon[example_i], (42, 42)),
            cmap='Greys_r', vmin=0, vmax=1, interpolation='none')
        axs[1][example_i].xaxis.set_visible(False)
        axs[1][example_i].yaxis.set_visible(False)

    fig.show()
    plt.show()

    recon = ae.generate(sess, np.random.randint(0, 2, (n_examples, 32)))
    fig, axs = plt.subplots(2, n_examples, figsize=(20, 4))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(atari_dataset['x'][example_i], (42, 42)),
            cmap='Greys_r', vmin=0, vmax=1, interpolation='none')
        axs[0][example_i].xaxis.set_visible(False)
        axs[0][example_i].yaxis.set_visible(False)
        axs[1][example_i].imshow(
            np.reshape(recon[example_i], (42, 42)),
            cmap='Greys_r', vmin=0, vmax=1, interpolation='none')
        axs[1][example_i].xaxis.set_visible(False)
        axs[1][example_i].yaxis.set_visible(False)

    tf.train.SummaryWriter('/Users/rein/programming/tensorboard/logs', sess.graph)
    fig.show()
    plt.show()


if __name__ == '__main__':
    test_atari()
