from contextlib import contextmanager

from rllab import config
import os
from rllab.misc import console
from rllab.misc import logger
import urllib.request
import tensorflow as tf
import numpy as np
import pyprind
from sandbox.rocky.gm.addict import Dict
from sandbox.rocky.s3.resource_manager import resource_manager


def load_binary_mnist():
    mnist_dir = os.path.join(config.PROJECT_PATH, "data/resource/MNIST_data")
    console.mkdir_p(mnist_dir)
    sets = ['train', 'valid', 'test']
    ret = Dict()
    for set in sets:
        url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(set)
        resource_name = "binary_mnist/{}.npz".format(set)

        def mkfile():
            file_name = url.split('/')[-1]
            npz_file = file_name.replace('amat', 'npz')
            local_npz_file = os.path.join(mnist_dir, npz_file)
            if not os.path.exists(local_npz_file):
                local_amat_file = os.path.join(mnist_dir, file_name)
                if not os.path.exists(local_amat_file):
                    logger.log("Downloading %s" % file_name)
                    urllib.request.urlretrieve(url, local_amat_file)
                logger.log("Converting %s to %s" % (file_name, npz_file))
                data = np.cast['uint8'](np.loadtxt(local_amat_file))
                np.savez_compressed(local_npz_file, data)
            resource_manager.register_file(resource_name, local_npz_file)

        local_npz_file = resource_manager.get_file(resource_name, mkfile)
        ret[set] = np.load(local_npz_file)['arr_0'].reshape((-1, 28, 28)).astype(np.float32)
    return ret


TRAIN = 'train'
TEST = 'test'
PHASE = TRAIN


def is_training():
    return PHASE == TRAIN


@contextmanager
def training():
    global PHASE
    prev_phase = PHASE
    PHASE = TRAIN
    yield
    PHASE = prev_phase


@contextmanager
def testing():
    global PHASE
    prev_phase = PHASE
    PHASE = TEST
    yield
    PHASE = prev_phase


    # pass


def batch_norm(x, decay=0.999):
    with tf.variable_scope("bn"):
        x_shape = x.get_shape().as_list()
        rank = len(x_shape)
        var_shape = (1,) * (rank - 1) + (x_shape[-1],)
        moving_mean = tf.get_variable(
            "moving_mean",
            shape=var_shape,
            initializer=tf.zeros_initializer,
        )
        moving_variance = tf.get_variable(
            "moving_variance",
            shape=var_shape,
            initializer=tf.ones_initializer,
        )
        beta = tf.get_variable(
            "beta",
            shape=var_shape,
            initializer=tf.zeros_initializer,
        )
        gamma = tf.get_variable(
            "gamma",
            shape=var_shape,
            initializer=tf.ones_initializer,
        )
        if is_training():
            # compute moving average of existing data
            mean, variance = tf.nn.moments(x, axes=list(range(rank - 1)), keep_dims=True)
            updated_moving_mean = moving_mean * decay + mean * (1 - decay)
            updated_moving_variance = moving_variance * decay + variance * (1 - decay)
            with tf.control_dependencies([
                tf.assign(moving_mean, updated_moving_mean),
                tf.assign(moving_variance, updated_moving_variance),
            ]):
                return tf.nn.batch_normalization(x, updated_moving_mean, updated_moving_variance, offset=beta,
                                                 scale=gamma, variance_epsilon=0.001)
        else:
            # use computed mean and variance
            return tf.nn.batch_normalization(x, moving_mean, moving_variance, offset=beta,
                                             scale=gamma, variance_epsilon=0.001)


def conv2d(
        input,
        name,
        filter_size=(3, 3),
        num_filters=16,
        strides=(1, 1),
        padding='SAME',
        use_batch_norm=False,
):
    with tf.variable_scope(name):
        assert len(input.get_shape().as_list()) == 4
        in_channels = input.get_shape().as_list()[-1]
        filter = tf.get_variable(
            "filter",
            shape=filter_size + (in_channels, num_filters),
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )
        conv_result = tf.nn.conv2d(
            input=input,
            filter=filter,
            strides=(1, strides[0], strides[1], 1),
            padding=padding
        )
        if use_batch_norm:
            # bias is not needed
            return batch_norm(conv_result)
        else:
            bias = tf.get_variable(
                "bias",
                shape=(1, 1, 1, num_filters),
                dtype=tf.float32,
                initializer=tf.zeros_initializer,
            )
            return conv_result + bias


class DiagonalGaussian(object):
    def __init__(self, means, log_stds):
        self.means = means
        self.log_stds = log_stds

    def sample_sym(self):
        z = tf.random_normal(shape=tf.shape(self.means))
        return z * tf.exp(self.log_stds) + self.means

    def kl_sym(self, other):
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices
        """
        stds = tf.exp(self.log_stds)
        other_stds = tf.exp(other.log_stds)
        numerator = tf.square(self.means - other.means) + tf.square(stds) - tf.square(other_stds)
        denominator = 2 * tf.square(other_stds) + 1e-8
        return tf.reduce_sum(numerator / denominator + other.log_stds - self.log_stds, -1)


class Bernoulli(object):
    def __init__(self, logits):
        self.logits = logits

    def log_likelihood_sym(self, x_var):
        # p = tf.nn.sigmoid(self.logits)
        ndims = len(x_var.get_shape().as_list())
        return tf.reduce_sum(
            # x_var * tf.log(p) + (1 - x_var) * tf.log(1 - p),
            # -1
            -tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, targets=x_var),
            reduction_indices=list(range(1, ndims)),  # -1
        )


def linear(input, name, num_units, use_batch_norm=False):
    with tf.variable_scope(name):
        input_shape = input.get_shape().as_list()
        input_dim = np.prod(input_shape[1:])
        input = tf.reshape(input, (-1, input_dim))
        weight = tf.get_variable(
            "weight",
            shape=(input_dim, num_units),
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )
        prod = tf.matmul(input, weight)
        if use_batch_norm:
            return batch_norm(prod)
        else:
            bias = tf.get_variable(
                "bias",
                shape=(1, num_units),
                dtype=tf.float32,
                initializer=tf.zeros_initializer,
            )
            return prod + bias


def deconv2d(
        input, name, output_size, filter_size=(3, 3), num_filters=16, strides=(1, 1), padding='SAME',
        use_batch_norm=False,
):
    with tf.variable_scope(name):
        input_shape = input.get_shape().as_list()
        assert len(input_shape) == 4
        batch_size, in_height, in_width, in_channels = input_shape
        filter = tf.get_variable(
            "filter",
            shape=filter_size + (num_filters, in_channels),
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )
        deconv_result = tf.nn.conv2d_transpose(
            value=input,
            filter=filter,
            output_shape=(batch_size,) + output_size + (num_filters,),
            strides=(1, strides[0], strides[1], 1),
            padding=padding,
        )
        if use_batch_norm:
            return batch_norm(deconv_result)
        else:
            bias = tf.get_variable(
                "bias",
                shape=(1, 1, 1, num_filters),
                dtype=tf.float32,
                initializer=tf.zeros_initializer,
            )
            return deconv_result + bias


def optimize(
        input_vars,
        inputs,
        test_inputs,
        loss,
        diagnostics=None,
        learning_rate=1e-3,
        batch_size=128,
        optimizer="adam",
        n_epochs=100,
):
    with tf.Session() as sess:
        if optimizer == "adam":
            tf_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            raise NotImplementedError
        if diagnostics is None:
            diagnostics = []

        train_op = tf_optimizer.minimize(loss, var_list=tf.trainable_variables())
        diag_vars = [var for (key, var) in diagnostics]

        aggr_diag_vals = [[] for _ in diagnostics]
        test_aggr_diag_vals = [[] for _ in diagnostics]

        N = inputs[0].shape[0]
        test_N = test_inputs[0].shape[0]

        sess.run(tf.initialize_all_variables())

        for epoch_idx in range(n_epochs):
            logger.log("Starting epoch {}...".format(epoch_idx))
            ids = np.arange(N)
            np.random.shuffle(ids)
            progbar = pyprind.ProgBar(iterations=N)
            for batch_idx in range(0, N, batch_size):
                batch_ids = ids[batch_idx:batch_idx + batch_size]
                if len(batch_ids) != batch_size:
                    break
                batch_inputs = [x[batch_ids] for x in inputs]
                _, *diag_vals = sess.run(
                    [train_op] + diag_vars,
                    feed_dict=dict(zip(input_vars, batch_inputs))
                )
                for aggr, val in zip(aggr_diag_vals, diag_vals):
                    aggr.append(val)
                progbar.update(iterations=batch_size)
            if progbar.active:
                progbar.stop()
            logger.log("Evaluating on test set...")

            ids = np.arange(test_N)
            np.random.shuffle(ids)
            for batch_idx in range(0, test_N, batch_size):
                batch_ids = ids[batch_idx:batch_idx + batch_size]
                if len(batch_ids) != batch_size:
                    break
                batch_inputs = [x[batch_ids] for x in test_inputs]
                diag_vals = sess.run(
                    diag_vars,
                    feed_dict=dict(zip(input_vars, batch_inputs))
                )
                for aggr, val in zip(test_aggr_diag_vals, diag_vals):
                    aggr.append(val)

            logger.record_tabular('Epoch', epoch_idx)
            for (key, _), vals in zip(diagnostics, aggr_diag_vals):
                logger.record_tabular(key, np.mean(vals))
            for (key, _), vals in zip(diagnostics, test_aggr_diag_vals):
                logger.record_tabular('Test' + key, np.mean(vals))
            logger.dump_tabular()


if __name__ == "__main__":
    # imagine that we now have some image data

    tf.set_random_seed(0)

    mnist = load_binary_mnist()

    batch_size = 256  # 128
    image_size = 28
    latent_dim = 128

    image_var = tf.placeholder(dtype=tf.float32, shape=(batch_size, image_size, image_size, 1), name="image")

    use_batch_norm = False#True


    def encode(image_var, latent_dim):
        with tf.variable_scope("encoder"):
            l = image_var
            l = tf.nn.relu(
                conv2d(
                    l, name="conv1", filter_size=(3, 3), strides=(2, 2), num_filters=32,
                    use_batch_norm=use_batch_norm,
                )
            )
            l = tf.nn.relu(
                conv2d(
                    l, name="conv2", filter_size=(3, 3), strides=(1, 1), num_filters=32,
                    use_batch_norm=use_batch_norm,
                )
            )
            l = tf.nn.relu(
                conv2d(
                    l, name="conv3", filter_size=(3, 3), strides=(2, 2), num_filters=64,
                    use_batch_norm=use_batch_norm,
                )
            )
            l = tf.nn.relu(
                conv2d(
                    l, name="conv4", filter_size=(3, 3), strides=(1, 1), num_filters=64,
                    use_batch_norm=use_batch_norm,
                )
            )
            l = linear(l, name="fc1", num_units=latent_dim * 2)
            means, log_stds = tf.split(split_dim=1, num_split=2, value=l)
            return DiagonalGaussian(means=means, log_stds=log_stds)


    def decode(latent_var):
        with tf.variable_scope("decoder"):
            l = latent_var
            l = tf.nn.relu(
                linear(
                    l, name="fc1", num_units=7 * 7 * 64,
                    use_batch_norm=use_batch_norm,
                )
            )
            l = tf.reshape(l, (-1, 7, 7, 64))
            l = tf.nn.relu(
                deconv2d(
                    l, name="deconv1", output_size=(7, 7), filter_size=(3, 3), strides=(1, 1), num_filters=64,
                    use_batch_norm=use_batch_norm,
                )
            )
            l = tf.nn.relu(
                deconv2d(
                    l, name="deconv2", output_size=(14, 14), filter_size=(3, 3), strides=(2, 2), num_filters=32,
                    use_batch_norm=use_batch_norm,
                )
            )
            l = tf.nn.relu(
                deconv2d(
                    l, name="deconv3", output_size=(14, 14), filter_size=(3, 3), strides=(1, 1), num_filters=32,
                    use_batch_norm=use_batch_norm,
                )
            )
            l = deconv2d(
                l, name="deconv4", output_size=(28, 28), filter_size=(3, 3), strides=(2, 2), num_filters=1
            )
            return Bernoulli(logits=l)


    # TODO
    # Implement weight-norm
    # Is it possible to implement these as post-processing units?
    # def compute_total_cost

    encoder_dist = encode(image_var, latent_dim=latent_dim)
    decoder_dist = decode(encoder_dist.sample_sym())
    prior = DiagonalGaussian(
        means=tf.zeros((batch_size, latent_dim)),
        log_stds=tf.zeros((batch_size, latent_dim)),
    )

    reconstr_cost = -tf.reduce_mean(decoder_dist.log_likelihood_sym(image_var))
    kl_cost = tf.reduce_mean(encoder_dist.kl_sym(prior))

    total_cost = reconstr_cost + kl_cost

    optimize(
        input_vars=[image_var],
        inputs=[mnist.train.reshape((-1, 28, 28, 1))],
        test_inputs=[mnist.test.reshape((-1, 28, 28, 1))],
        loss=total_cost,
        diagnostics=[
            ("Loss", total_cost),
            ("VLB", -total_cost),
            ("ReconstrCost", reconstr_cost),
            ("KLCost", kl_cost),
        ],
        learning_rate=1e-3,
        batch_size=batch_size,
        optimizer="adam",
        n_epochs=10000,
    )
