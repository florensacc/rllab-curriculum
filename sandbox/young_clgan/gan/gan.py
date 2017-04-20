import copy
from rllab.misc import logger

import numpy as np

import tensorflow as tf
import tflearn

DEFAULT_GAN_CONFIGS = {
    'batch_size': 64,
    'generator_activation': 'relu',
    'discriminator_activation': 'relu',
    'generator_optimizer': tf.train.AdamOptimizer(0.001),
    'discriminator_optimizer': tf.train.AdamOptimizer(0.001),
    'generator_weight_initializer': tflearn.initializations.truncated_normal(stddev=2),
    'discriminator_weight_initializer': tflearn.initializations.truncated_normal(stddev=2),
    'print_iteration': 20,
    'reset_generator_optimizer': True,
    'reset_discriminator_optimizer': True,
    'batch_normalize_discriminator': True,
    'discriminator_batch_noise_stddev': 0,
    'generator_loss_weights': None,
}


class FCGAN(object):
    def __init__(self, generator_output_size, discriminator_output_size,
                 generator_layers, discriminator_layers, noise_size, tf_session,
                 discriminator_max_iters=200, generator_max_iters=10, outer_iters=5,
                 discriminator_min_loss=-np.infty, generator_min_loss=-np.infty,
                 configs=None):

        self.noise_size = noise_size
        self.tf_graph = tf.Graph()
        self.tf_session = tf_session
        self.configs = copy.deepcopy(DEFAULT_GAN_CONFIGS)
        # training hyperparams
        self.outer_iters = outer_iters
        self.discriminator_max_iters = discriminator_max_iters
        self.generator_max_iters = generator_max_iters
        self.discriminator_min_loss = discriminator_min_loss
        self.generator_min_loss = generator_min_loss
        if configs is not None:
            self.configs.update(configs)

        # with self.tf_graph.as_default():
        with tf.variable_scope("generator"):
            self.generator = Generator(
                generator_output_size, generator_layers, noise_size,
                self.configs,
            )

        with tf.variable_scope("discriminator"):
            self.discriminator = Discriminator(
                self.generator.output, generator_output_size,
                discriminator_layers, discriminator_output_size, self.configs,
            )

        self.generator_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            'generator'
        )

        self.discriminator_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            'discriminator'
        )

        with tf.variable_scope('fcgan_generator_optimizer'):
            self.generator_train_op = configs['generator_optimizer'].minimize(
                self.discriminator.generator_loss,
                var_list=self.generator_variables
            )
        with tf.variable_scope('fcgan_discriminator_optimizer'):
            self.discriminator_train_op = configs['discriminator_optimizer'].minimize(
                self.discriminator.discriminator_loss,
                var_list=self.discriminator_variables
            )

        self.generator_optimizer_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            'fcgan_generator_optimizer'
        )

        self.discriminator_optimizer_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            'fcgan_discriminator_optimizer'
        )

        self.initialize_trainable_variable_op = tf.variables_initializer(
            (self.generator_variables
             + self.discriminator_variables
             + self.generator_optimizer_variables
             + self.discriminator_optimizer_variables)
        )
        self.tf_session.run(
            self.initialize_trainable_variable_op
        )

        self.initialize_generator_optimizer_op = tf.variables_initializer(
            self.generator_optimizer_variables
        )

        self.initialize_discriminator_optimizer_op = tf.variables_initializer(
            self.discriminator_optimizer_variables
        )

        self.tf_session.run(
            self.initialize_generator_optimizer_op
        )
        self.tf_session.run(
            self.initialize_discriminator_optimizer_op
        )

    def initialize(self):
        self.tf_session.run(
            self.initialize_trainable_variable_op
        )
        self.tf_session.run(
            self.initialize_generator_optimizer_op
        )
        self.tf_session.run(
            self.initialize_discriminator_optimizer_op
        )

    def sample_random_noise(self, size):
        generator_noise = []
        batch_size = self.configs['batch_size']
        for i in range(0, size, batch_size):
            sample_size = min(batch_size, size - i)
            noise = np.random.randn(sample_size, self.noise_size)
            generator_noise.append(noise)

        return np.vstack(generator_noise)

    def sample_generator(self, size):
        tflearn.config.is_training(is_training=False, session=self.tf_session)
        generator_samples = []
        generator_noise = []
        batch_size = self.configs['batch_size']
        for i in range(0, size, batch_size):
            sample_size = min(batch_size, size - i)
            noise = np.random.randn(sample_size, self.noise_size)
            generator_noise.append(noise)
            generator_samples.append(
                self.tf_session.run(
                    self.generator.output,
                    {self.generator.input: noise}
                )
            )
        return np.vstack(generator_samples), np.vstack(generator_noise)

    def train(self, X, Y, outer_iters=None, generator_max_iters=None, discriminator_max_iters=None,
              suppress_generated_states=True, discriminator_min_loss=None, generator_min_loss=None):

        # overwrite the ones in init if provided
        outer_iters = self.outer_iters if outer_iters is None else outer_iters
        generator_max_iters = self.generator_max_iters if generator_max_iters is None else generator_max_iters
        discriminator_max_iters = self.discriminator_max_iters if discriminator_max_iters is None else discriminator_max_iters
        discriminator_min_loss = self.discriminator_min_loss if discriminator_min_loss is None else discriminator_min_loss
        generator_min_loss = self.generator_min_loss if generator_min_loss is None else generator_min_loss

        sample_size = X.shape[0]
        train_size = min(
            int(self.configs['batch_size'] * discriminator_max_iters / 10),
            sample_size
        )
        generated_Y = np.zeros((train_size, Y.shape[1]))
        for i in range(outer_iters):
            print("\n*******  GAN iter {} ******".format(i))
            if self.configs['reset_generator_optimizer']:
                self.tf_session.run(
                    self.initialize_generator_optimizer_op
                )
            if self.configs['reset_discriminator_optimizer']:
                self.tf_session.run(
                    self.initialize_discriminator_optimizer_op
                )

            indices = np.random.randint(0, X.shape[0], size=(train_size,))
            sample_X = X[indices, :]
            sample_Y = Y[indices, :]

            if suppress_generated_states:
                generated_X, random_noise = self.sample_generator(train_size)
                feed_X = np.vstack([sample_X, generated_X])
                feed_Y = np.vstack([sample_Y, generated_Y])
            else:
                random_noise = self.sample_random_noise(train_size)
                feed_X = np.vstack([sample_X])
                feed_Y = np.vstack([sample_Y])

            dis_log_loss = self.train_discriminator(feed_X, feed_Y, discriminator_max_iters, discriminator_min_loss)
            gen_log_loss = self.train_generator(random_noise, generator_max_iters, generator_min_loss)
        # if i == 0:
        #         logger.record_tabular('Discrim_lossBefore', dis_log_loss[0])
        #         logger.record_tabular('Gen_lossBefore', gen_log_loss[0])
        # logger.record_tabular('Discim_lossAfter', dis_log_loss[-1])
        # logger.record_tabular('Gen_lossAfter', gen_log_loss[-1])
        return dis_log_loss, gen_log_loss

    def train_discriminator(self, X, Y, max_iters, min_loss=-np.infty):
        """
        :param X: states that we know lables of
        :param Y: labels of those states
        :param max_iters: of the discriminator trainig
        :param min_loss: beyond this loss we don't keep training
        The batch size is given by the configs of the class!
        discriminator_batch_noise_stddev > 0: check that std on each component is at least this. (if com: 2)
        """
        log_loss = []
        tflearn.config.is_training(is_training=True, session=self.tf_session)
        batch_size = self.configs['batch_size']
        for i in range(max_iters):
            indices = np.random.randint(0, X.shape[0], size=(batch_size,))
            train_X = X[indices, :]
            train_Y = Y[indices]

            if self.configs['discriminator_batch_noise_stddev'] > 0:
                noise_indices = np.var(train_X, axis=0) < self.configs['discriminator_batch_noise_stddev']
                noise = np.random.randn(*train_X.shape)
                noise[:, np.logical_not(noise_indices)] = 0
                train_X += noise * self.configs['discriminator_batch_noise_stddev']

            self.tf_session.run(
                self.discriminator_train_op,
                {self.discriminator.sample_input: train_X,
                 self.discriminator.label: train_Y}
            )

            if i % self.configs['print_iteration'] == 0 or i == max_iters - 1:
                loss = self.tf_session.run(
                    self.discriminator.discriminator_loss,
                    {self.discriminator.sample_input: train_X,
                     self.discriminator.label: train_Y}
                )
                log_loss.append(loss)
                print('Disc_itr_%i: discriminator loss: %f' % (i, loss))
                if loss < min_loss:
                    break
        return log_loss  # the length of this already tells the number of itrs that were used

    def train_generator(self, X, max_iters, min_loss=-np.infty):
        """
        :param X: These are the latent variables that were used to generate??
        :param max_iters: maximum number of itrs
        :param min_loss: beyond this loss we don't keep training
        :return:
        """
        log_loss = []
        tflearn.config.is_training(is_training=False, session=self.tf_session)
        batch_size = self.configs['batch_size']
        for i in range(max_iters):
            indices = np.random.randint(0, X.shape[0], size=(batch_size,))
            train_X = X[indices, :]

            self.tf_session.run(
                self.generator_train_op,
                {self.generator.input: train_X}
            )

            if i % self.configs['print_iteration'] == 0 or i == max_iters - 1:
                loss = self.tf_session.run(
                    self.discriminator.generator_loss,
                    {self.generator.input: train_X}
                )
                log_loss.append(loss)
                print('Gen_itr_%i: generator loss: %f' % (i, loss))
                if loss < min_loss:
                    break
        return log_loss

    def discriminator_predict(self, X):
        tflearn.config.is_training(is_training=False, session=self.tf_session)
        batch_size = self.configs['batch_size']
        output = []
        for i in range(0, X.shape[0], batch_size):
            sample_size = min(batch_size, X.shape[0] - i)
            output.append(
                self.tf_session.run(
                    self.discriminator.output,
                    {self.discriminator.sample_input: X[i:i + sample_size]}
                )
            )
        return np.vstack(output)


class Generator(object):
    def __init__(self, output_size, hidden_layers, noise_size, configs):
        self.configs = configs
        self._input = tf.placeholder(tf.float32, shape=[None, noise_size])
        out = self.input

        if callable(configs['generator_activation']):
            activation = configs['generator_activation']()
        else:
            activation = configs['generator_activation']

        for size in hidden_layers:
            out = tflearn.fully_connected(
                out, size, activation=activation, regularizer=None,
                weights_init=configs['generator_weight_initializer'],
            )
        self._output = tflearn.fully_connected(
            out, output_size, activation='tanh', regularizer=None,
            weights_init=configs['generator_weight_initializer'],
        )

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output


class Discriminator(object):
    def __init__(self, generator_output, input_size,
                 hidden_layers, output_size, configs):
        self._generator_input = generator_output
        self._sample_input = tf.placeholder(tf.float32, shape=[None, input_size])
        self._label = tf.placeholder(tf.float32, shape=[None, output_size])
        self.configs = configs
        # we initialize??
        generator_out = self._generator_input
        sample_out = self._sample_input

        if callable(configs['discriminator_activation']):
            activation = configs['discriminator_activation']()
        else:
            activation = configs['discriminator_activation']
        for i, size in enumerate(hidden_layers):
            generator_out = tflearn.fully_connected(
                generator_out, size, activation=activation, regularizer=None,
                scope='fc_{}'.format(i),
                weights_init=configs['discriminator_weight_initializer'],
            )
            if configs['batch_normalize_discriminator']:
                generator_out = tflearn.layers.normalization.batch_normalization(
                    generator_out, scope='fc_{}'.format(i)
                )
            sample_out = tflearn.fully_connected(
                sample_out, size, activation=activation, regularizer=None,
                scope='fc_{}'.format(i), reuse=True
            )
            if configs['batch_normalize_discriminator']:
                sample_out = tflearn.layers.normalization.batch_normalization(
                    sample_out, scope='fc_{}'.format(i), reuse=True
                )

        self._generator_output = tflearn.fully_connected(
            generator_out, output_size, activation='sigmoid', regularizer=None,
            scope='fc_out'.format(i),
            weights_init=configs['discriminator_weight_initializer'],
        )
        self._sample_output = tflearn.fully_connected(
            sample_out, output_size, activation='sigmoid', regularizer=None,
            scope='fc_out'.format(i), reuse=True
        )

        self._discriminator_loss = tf.reduce_mean(
            -tf.reduce_sum(
                self._label * tf.log(self._sample_output + 1e-10) + (1 - self._label) * tf.log(
                    1 - self._sample_output + 1e-10),
                reduction_indices=[1]
            )
        )

        generator_loss_weights = configs['generator_loss_weights']
        if generator_loss_weights is None:
            generator_loss_weights = 1
        else:
            generator_loss_weights = np.array(generator_loss_weights).reshape(1, -1)

        self._generator_loss = tf.reduce_mean(
            -tf.reduce_sum(
                tf.log(self._generator_output + 1e-10) * generator_loss_weights,
                reduction_indices=[1]
            )
        )

    @property
    def sample_input(self):
        return self._sample_input

    @property
    def generator_input(self):
        return self._generator_input

    @property
    def generator_output(self):
        return self._generator_output

    @property
    def sample_output(self):
        return self._sample_output

    @property
    def label(self):
        return self._label

    @property
    def discriminator_loss(self):
        return self._discriminator_loss

    @property
    def generator_loss(self):
        return self._generator_loss
