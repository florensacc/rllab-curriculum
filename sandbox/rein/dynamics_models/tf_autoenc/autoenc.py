import tensorflow as tf
import numpy as np
from sandbox.rein.dynamics_models.utils import load_dataset_atari
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.misc.tensor_utils import to_onehot_sym

# --
# Nonscientific printing of numpy arrays.
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)


class IndependentSoftmaxLayer(L.Layer):
    def __init__(self, incoming, num_bins, W=L.XavierUniformInitializer(), b=tf.zeros_initializer,
                 **kwargs):
        super(IndependentSoftmaxLayer, self).__init__(incoming, **kwargs)

        self._num_bins = num_bins
        self.W = self.add_param(W, (self.input_shape[3], self._num_bins), name='W')
        self.b = self.add_param(b, (self._num_bins,), name='b')
        self.pixel_b = self.add_param(
            b,
            (self.input_shape[1], self.input_shape[2], self._num_bins,),
            name='pixel_b'
        )

    def get_output_for(self, input, **kwargs):
        shape = (-1, self.input_shape[3])
        input_reshaped = tf.reshape(input, shape)
        in_dot_w = tf.matmul(input_reshaped, self.W) + tf.expand_dims(self.b, 0)
        shp = (-1, self.input_shape[1], self.input_shape[2], self._num_bins)
        fc_biased = tf.reshape(in_dot_w, shp) + self.pixel_b
        fc_biased_rshp = tf.reshape(fc_biased, [-1, self._num_bins])
        out = tf.nn.softmax(fc_biased_rshp)
        return tf.reshape(out, shp)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], self._num_bins


class BinaryCodeNonlinearityLayer(L.Layer):
    """
    Discrete embedding layer, the nonlinear part
    This has to be put after the batch norm layer.
    """

    def __init__(self, incoming, num_units,
                 **kwargs):
        super(BinaryCodeNonlinearityLayer, self).__init__(incoming, **kwargs)
        self.num_units = num_units

    @staticmethod
    def nonlinearity(x, noise_mask=1):
        # Force outputs to be binary through noise.
        return tf.nn.sigmoid(x) + noise_mask * tf.random_uniform(shape=tf.shape(x), minval=-0.3, maxval=0.3)

    # FIXME: noise_mask = 0
    def get_output_for(self, input, noise_mask=0, **kwargs):
        return self.nonlinearity(input, noise_mask)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.num_units


class BinaryCodeConvAE:
    """Convolutional/Deconvolutional autoencoder with shared weights.
    """

    def __init__(self,
                 input_shape=(42, 42, 1),
                 label_smoothing=0,
                 num_softmax_bins=64,
                 code_dimension=32,
                 ):
        self._label_smoothing = label_smoothing
        self._num_softmax_bins = num_softmax_bins
        self._code_dimension = code_dimension

        self._x = tf.placeholder(tf.float32, shape=(None,) + input_shape, name="input")
        l_in = L.InputLayer(shape=(None,) + input_shape, input_var=self._x, name="input_layer")
        l_conv_1 = L.Conv2DLayer(
            l_in,
            num_filters=96,
            filter_size=5,
            stride=(2, 2),
            pad='VALID',
            nonlinearity=tf.nn.relu,
            name='enc_conv_1',
            weight_normalization=True,
        )
        l_conv_1 = L.batch_norm(l_conv_1)
        l_conv_2 = L.Conv2DLayer(
            l_conv_1,
            num_filters=96,
            filter_size=5,
            stride=(2, 2),
            pad='VALID',
            nonlinearity=tf.nn.relu,
            name='enc_conv_2',
            weight_normalization=True,
        )
        l_conv_2 = L.batch_norm(l_conv_2)
        l_flatten_1 = L.FlattenLayer(l_conv_2)
        l_dense_1 = L.DenseLayer(
            l_flatten_1,
            num_units=128,
            nonlinearity=tf.nn.relu,
            name='enc_hidden_1',
            W=L.XavierUniformInitializer(),
            b=tf.zeros_initializer,
            weight_normalization=True
        )
        l_dense_1 = L.batch_norm(l_dense_1)
        l_code_prenoise = L.DenseLayer(
            l_dense_1,
            num_units=self._code_dimension,
            nonlinearity=tf.identity,
            name='binary_code_prenoise',
            W=L.XavierUniformInitializer(),
            b=tf.zeros_initializer,
            weight_normalization=True
        )
        l_code = BinaryCodeNonlinearityLayer(
            l_code_prenoise,
            num_units=self._code_dimension,
            name='binary_code',
        )
        l_dense_3 = L.DenseLayer(
            l_code,
            num_units=np.prod(l_conv_2.output_shape[1:]),
            nonlinearity=tf.nn.sigmoid,
            name='dec_hidden_1',
            W=L.XavierUniformInitializer(),
            b=tf.zeros_initializer,
            weight_normalization=True
        )
        l_dense_3 = L.batch_norm(l_dense_3)
        l_reshp_1 = L.ReshapeLayer(
            l_dense_3,
            (-1,) + l_conv_2.output_shape[1:]
        )
        l_deconv_1 = L.TransposedConv2DLayer(
            l_reshp_1,
            num_filters=96,
            filter_size=5,
            stride=(2, 2),
            W=L.XavierUniformInitializer(),
            b=tf.zeros_initializer,
            crop='VALID',
            nonlinearity=tf.nn.relu,
            name='dec_deconv_1',
            weight_normalization=True,
        )
        l_deconv_1 = L.batch_norm(l_deconv_1)
        l_deconv_2 = L.TransposedConv2DLayer(
            l_deconv_1,
            num_filters=1,
            filter_size=6,
            stride=(2, 2),
            W=L.XavierUniformInitializer(),
            b=tf.zeros_initializer,
            crop='VALID',
            nonlinearity=tf.nn.sigmoid,
            name='dec_deconv_2',
            weight_normalization=True,
        )
        l_softmax = IndependentSoftmaxLayer(
            l_deconv_2,
            num_bins=self._num_softmax_bins,
        )
        l_out = l_softmax
        print(l_out.output_shape)
        # l_out = l_deconv_2

        print(l_conv_1.output_shape)
        print(l_conv_2.output_shape)
        print(l_deconv_1.output_shape)
        print(l_deconv_2.output_shape)

        def likelihood_classification(target, prediction):
            def _log_prob_softmax_onehot(target, prediction):
                # Cross-entropy; target vector selecting correct prediction
                # entries.
                ll = tf.reduce_sum((
                    target * tf.log(prediction)
                ), 1)
                return ll

            target = tf.cast(to_onehot_sym(
                tf.cast(tf.reshape(target, [-1]), 'int32'),
                self._num_softmax_bins
            ), tf.float32)
            target += self._label_smoothing
            target = target / tf.reduce_sum(target, 1, keep_dims=True)
            return tf.reduce_sum(_log_prob_softmax_onehot(
                target,
                tf.reshape(prediction, [-1, self._num_softmax_bins])
            ))

        # --
        self._z = L.get_output(l_code, noise_mask=0)

        # --
        self._y = L.get_output(l_out, deterministic=True)

        # --
        self._z_in = tf.placeholder(tf.float32, shape=(None, self._code_dimension), name="z_in")
        self._y_gen = L.get_output(l_out, {l_code: self._z_in}, deterministic=True)

        # --
        self._t = tf.placeholder(tf.int32, shape=(None,) + input_shape, name="target")
        self._cost = - likelihood_classification(
            self._t, L.get_output(l_out)) / tf.cast(tf.shape(self._x)[0], tf.float32)

        # --
        learning_rate = 0.01
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
            z = np.random.randint(0, 2, (1, self._code_dimension))
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        softmax_out = sess.run(self._y_gen, feed_dict={self._z_in: z})
        return np.argmax(softmax_out, axis=-1) / float(self._num_softmax_bins - 1)

    def reconstruct(self, sess, X):
        """ Use VAE to reconstruct given data. """
        softmax_out = sess.run(self.y, feed_dict={self.x: X})
        return np.argmax(softmax_out, axis=-1) / float(self._num_softmax_bins - 1)

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
    num_bins = 64
    bin_code_dim = 32
    atari_dataset = load_dataset_atari('/Users/rein/programming/datasets/dataset_42x42.pkl')
    atari_dataset['x'] = atari_dataset['x'].transpose((0, 2, 3, 1))
    atari_dataset['y'] = (atari_dataset['x'] * (num_bins - 1)).astype(np.int)
    ae = BinaryCodeConvAE(label_smoothing=0.003, num_softmax_bins=num_bins)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    n_epochs = 1000
    for epoch_i in range(n_epochs):
        train_x = atari_dataset['x']
        train_y = atari_dataset['y']
        sess.run(ae.optimizer, feed_dict={ae.x: train_x, ae._t: train_y})
        print('{:3d}'.format(epoch_i), sess.run(ae.cost, feed_dict={ae.x: train_x, ae._t: train_y}))

    n_examples = 10
    # examples = np.tile(atari_dataset['x'][0][None, :], (n_examples, 1, 1, 1))
    examples = atari_dataset['x'][0:n_examples]

    recon = ae.reconstruct(sess, examples)
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

    recon = ae.generate(sess, np.random.randint(0, 2, (n_examples, bin_code_dim)))
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
