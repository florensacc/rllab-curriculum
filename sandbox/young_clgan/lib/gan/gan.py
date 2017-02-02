import copy

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
                 generator_layers, discriminator_layers, noise_size,
                 tf_session, configs=None):

        self.noise_size = noise_size
        self.tf_graph = tf.Graph()
        self.tf_session = tf_session
        self.configs = copy.deepcopy(DEFAULT_GAN_CONFIGS)
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
            tf.GraphKeys.VARIABLES,
            'generator'
        )

        self.discriminator_variables = tf.get_collection(
            tf.GraphKeys.VARIABLES,
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
            tf.GraphKeys.VARIABLES,
            'fcgan_generator_optimizer'
        )

        self.discriminator_optimizer_variables = tf.get_collection(
            tf.GraphKeys.VARIABLES,
            'fcgan_discriminator_optimizer'
        )

        self.initialize_trainable_variable_op = tf.initialize_variables(
            (self.generator_variables
             + self.discriminator_variables
             + self.generator_optimizer_variables
             + self.discriminator_optimizer_variables)
        )
        self.tf_session.run(
            self.initialize_trainable_variable_op
        )

        self.initialize_generator_optimizer_op = tf.initialize_variables(
            self.generator_optimizer_variables
        )

        self.initialize_discriminator_optimizer_op = tf.initialize_variables(
            self.discriminator_optimizer_variables
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

    def train(self, X, Y, outer_iters, generator_iters, discriminator_iters, suppress_generated_goals=False):
        sample_size = X.shape[0]
        train_size = min(
            int(self.configs['batch_size'] * discriminator_iters / 10),
            sample_size
        )
        generated_Y = np.zeros((train_size, Y.shape[1]))
        for _ in range(outer_iters):
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

            if suppress_generated_goals:
                generated_X, random_noise = self.sample_generator(train_size)
                feed_X = np.vstack([sample_X, generated_X])
                feed_Y = np.vstack([sample_Y, generated_Y])
            else:
                random_noise = self.sample_random_noise(train_size)
                feed_X = np.vstack([sample_X])
                feed_Y = np.vstack([sample_Y])

            self.train_discriminator(feed_X, feed_Y, discriminator_iters)
            self.train_generator(random_noise, generator_iters)

    def train_discriminator(self, X, Y, iters):
        """
        :param X: goal that we know lables of
        :param Y: labels of those goals
        :param iters: of the discriminator trainig
        The batch size is given by the configs of the class!
        discriminator_batch_noise_stddev > 0: check that std on each component is at least this. (if com: 2)
        """
        tflearn.config.is_training(is_training=True, session=self.tf_session)
        batch_size = self.configs['batch_size']
        for i in range(iters):
            indices = np.random.randint(0, X.shape[0], size=(batch_size,))
            train_X = X[indices, :]
            train_Y = Y[indices]

            if self.configs['discriminator_batch_noise_stddev'] > 0:
                noise_indices = np.var(train_X, axis=0) < self.configs['discriminator_batch_noise_stddev']
                noise = np.random.randn(*train_X.shape)
                noise[np.logical_not(noise_indices)] = 0
                train_X += noise * self.configs['discriminator_batch_noise_stddev']

            self.tf_session.run(
                self.discriminator_train_op,
                {self.discriminator.sample_input: train_X,
                 self.discriminator.label: train_Y}
            )

            if i % self.configs['print_iteration'] == 0:
                loss = self.tf_session.run(
                    self.discriminator.discriminator_loss,
                    {self.discriminator.sample_input: train_X,
                     self.discriminator.label: train_Y}
                )
                print('discriminator loss: %f' % loss)
                # if loss < 1e-3:
                #     break

    def train_generator(self, X, iters):
        """
        :param X: These are the latent variables that were used to generate??
        :param iters:
        :return:
        """
        tflearn.config.is_training(is_training=False, session=self.tf_session)
        batch_size = self.configs['batch_size']
        for i in range(iters):
            indices = np.random.randint(0, X.shape[0], size=(batch_size,))
            train_X = X[indices, :]

            self.tf_session.run(
                self.generator_train_op,
                {self.generator.input: train_X}
            )

            if i % self.configs['print_iteration'] == 0:
                loss = self.tf_session.run(
                    self.discriminator.generator_loss,
                    {self.generator.input: train_X}
                )
                print('generator loss: %f' % loss)
                # if loss < 1e-3:
                #     break

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
