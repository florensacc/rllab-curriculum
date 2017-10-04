import copy
from rllab.misc import logger

import numpy as np

import tensorflow as tf

# Here we use a function to avoid reuse of objects
DEFAULT_GAN_CONFIGS = lambda: {
    'batch_size': 64,
    'generator_output_activation': 'tanh',
    'generator_hidden_activation': 'relu',
    'discriminator_hidden_activation': 'leaky_relu',
    'generator_optimizer': tf.train.RMSPropOptimizer(0.001),
    'discriminator_optimizer': tf.train.RMSPropOptimizer(0.001),
    'generator_weight_initializer': tf.contrib.layers.xavier_initializer(),
    'discriminator_weight_initializer': tf.contrib.layers.xavier_initializer(),
    'print_iteration': 50,
    'reset_generator_optimizer': False,
    'reset_discriminator_optimizer': False,
    'batch_normalize_discriminator': False,
    'batch_normalize_generator': False,
    'supress_all_logging': False,
    'default_generator_iters': 1,   # It is highly recommend to not change these parameters
    'default_discriminator_iters': 1,
    'gan_type': 'lsgan',
    'wgan_gradient_penalty': 0.1,
}


def batch_feed_array(array, batch_size):
    data_size = array.shape[0]
    #assert data_size >= batch_size
    
    if data_size <= batch_size:
        while True:
            yield array
    else:
        start = 0
        while True:
            if start + batch_size < data_size:
                yield array[start:start + batch_size, ...]
            else:
                yield np.concatenate(
                    [array[start:data_size], array[0: start + batch_size - data_size]],
                    axis=0
                )
            start = (start + batch_size) % data_size


class FCGAN(object):
    def __init__(self, generator_output_size, discriminator_output_size,
                 generator_layers, discriminator_layers, noise_size,
                 tf_session, configs=None):

        self.generator_output_size = generator_output_size
        self.discriminator_output_size = discriminator_output_size
        self.noise_size = noise_size
        self.tf_graph = tf.Graph()
        self.tf_session = tf_session
        self.configs = copy.deepcopy(DEFAULT_GAN_CONFIGS())
        if configs is not None:
            self.configs.update(configs)
            
        self.generator_is_training = tf.placeholder_with_default(False, [])
        self.discriminator_is_training = tf.placeholder_with_default(False, [])

        # with self.tf_graph.as_default():
        with tf.variable_scope("generator"):
            self.generator = Generator(
                generator_output_size, generator_layers, noise_size,
                self.generator_is_training, self.configs,
            )

        with tf.variable_scope("discriminator"):
            self.discriminator = Discriminator(
                self.generator.output, generator_output_size,
                discriminator_layers, discriminator_output_size,
                self.discriminator_is_training, self.configs,
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
            self.generator_train_op = self.configs['generator_optimizer'].minimize(
                self.discriminator.generator_loss,
                var_list=self.generator_variables
            )
        with tf.variable_scope('fcgan_discriminator_optimizer'):
            self.discriminator_train_op = self.configs['discriminator_optimizer'].minimize(
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
        return np.random.randn(size, self.noise_size)

    def sample_generator(self, size):
        generator_samples = []
        generator_noise = []
        batch_size = self.configs['batch_size']
        for i in range(0, size, batch_size):
            sample_size = min(batch_size, size - i)
            noise = self.sample_random_noise(sample_size)
            generator_noise.append(noise)
            generator_samples.append(
                self.tf_session.run(
                    self.generator.output,
                    {self.generator.input: noise}
                )
            )
        return np.vstack(generator_samples), np.vstack(generator_noise)

    def train(self, X, Y, outer_iters, generator_iters=None, discriminator_iters=None):
        if generator_iters is None:
            generator_iters = self.configs['default_generator_iters']
        if discriminator_iters is None:
            discriminator_iters = self.configs['default_discriminator_iters']
        
        sample_size = X.shape[0]
        train_size = sample_size
        
        batch_size = self.configs['batch_size']
        
        generated_Y = np.zeros((batch_size, self.discriminator_output_size))
        
        batch_feed_X = batch_feed_array(X, batch_size)
        batch_feed_Y = batch_feed_array(Y, batch_size)
        
        
        for i in range(outer_iters):
            if self.configs['reset_generator_optimizer']:
                self.tf_session.run(
                    self.initialize_generator_optimizer_op
                )
            if self.configs['reset_discriminator_optimizer']:
                self.tf_session.run(
                    self.initialize_discriminator_optimizer_op
                )
            
            for j in range(discriminator_iters):
                sample_X = next(batch_feed_X)
                sample_Y = next(batch_feed_Y)
                generated_X, random_noise = self.sample_generator(batch_size)
                
                train_X = np.vstack([sample_X, generated_X])
                train_Y = np.vstack([sample_Y, generated_Y])
                
                dis_log_loss = self.train_discriminator(
                    train_X, train_Y, 1, no_batch=True
                )
            
            for i in range(generator_iters):
                gen_log_loss = self.train_generator(random_noise, 1)
                if i > 5:
                    random_noise = self.sample_random_noise(batch_size)
                
            
            if i % self.configs['print_iteration'] == 0 and not self.configs['supress_all_logging']:
                print('Iter: {}, generator loss: {}, discriminator loss: {}'.format(i, gen_log_loss, dis_log_loss))

        return dis_log_loss, gen_log_loss

    def train_discriminator(self, X, Y, iters, no_batch=False):
        """
        :param X: goal that we know lables of
        :param Y: labels of those goals
        :param iters: of the discriminator trainig
        The batch size is given by the configs of the class!
        discriminator_batch_noise_stddev > 0: check that std on each component is at least this. (if com: 2)
        """
        if no_batch:
            assert X.shape[0] == Y.shape[0]
            batch_size = X.shape[0]
        else:
            batch_size = self.configs['batch_size']
        
        batch_feed_X = batch_feed_array(X, batch_size)
        batch_feed_Y = batch_feed_array(Y, batch_size)
        
        for i in range(iters):
            train_X = next(batch_feed_X)
            train_Y = next(batch_feed_Y)

            loss, _ = self.tf_session.run(
                [self.discriminator.discriminator_loss, self.discriminator_train_op],
                {self.discriminator.sample_input: train_X,
                 self.discriminator.label: train_Y,
                 self.discriminator_is_training: True}
            )
                
        return loss

    def train_generator(self, X, iters):
        """
        :param X: These are the latent variables that were used to generate??
        :param iters:
        :return:
        """
        log_loss = []
        batch_size = self.configs['batch_size']
        
        batch_feed_X = batch_feed_array(X, batch_size)
        
        for i in range(iters):
            train_X = next(batch_feed_X)

            loss, _ = self.tf_session.run(
                [self.discriminator.generator_loss, self.generator_train_op],
                {self.generator.input: train_X, self.generator_is_training: True}
            )
               
        return loss

    def discriminator_predict(self, X):
        batch_size = self.configs['batch_size']
        output = []
        for i in range(0, X.shape[0], batch_size):
            sample_size = min(batch_size, X.shape[0] - i)
            output.append(
                self.tf_session.run(
                    self.discriminator.sample_output,
                    {self.discriminator.sample_input: X[i:i + sample_size]}
                )
            )
        return np.vstack(output)


class Generator(object):
    def __init__(self, output_size, hidden_layers, noise_size, is_training, configs):
        self.configs = configs
        self._input = tf.placeholder(tf.float32, shape=[None, noise_size])
        out = self._input

        for size in hidden_layers:
            out = tf.layers.dense(
                out, size,
                kernel_initializer=configs['generator_weight_initializer'],
            )
            
            if configs['generator_hidden_activation'] == 'relu':
                out = tf.nn.relu(out)
            elif configs['generator_hidden_activation'] == 'leaky_relu':
                out = tf.maximum(0.2 * out, out)
            else:
                raise ValueError('Unsupported activation type')
                
            if configs['batch_normalize_generator']:
                out = tf.layers.batch_normalization(
                    out, training=is_training
                )
            
        out = tf.layers.dense(
            out, output_size,
            kernel_initializer=configs['generator_weight_initializer'],
        )
        
        if configs['generator_output_activation'] == 'tanh':
            self._output = tf.nn.tanh(out)
        elif configs['generator_output_activation'] == 'sigmoid':
            self._output = tf.nn.sigmoid(out)
        elif configs['generator_output_activation'] == 'linear':
            self._output = out
        else:
            raise ValueError('Unsupported activation type!')
        

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output


class Discriminator(object):
    def __init__(self, generator_output, input_size, hidden_layers, output_size,
                 is_training, configs):
        self._generator_input = generator_output
        self._sample_input = tf.placeholder(tf.float32, shape=[None, input_size])
        self._label = tf.placeholder(tf.float32, shape=[None, output_size])
        self.configs = configs
        
        self.sample_discriminator = DiscriminatorNet(
            self._sample_input, hidden_layers, output_size, is_training,
            configs, reuse=False
        )
        
        self.generator_discriminator = DiscriminatorNet(
            self._generator_input, hidden_layers, output_size, is_training,
            configs, reuse=True
        )
        
        
        if configs['gan_type'] == 'wgan':
            self._generator_output = self.generator_discriminator.output
            self._sample_output = self.sample_discriminator.output
            
            self._discriminator_loss_logits = tf.reduce_mean(
                -2 * (self._label - 0.5) * self._sample_output
            )
            
            
            self._discriminator_loss_gradient = tf.nn.relu(
                tf.nn.l2_loss(
                    tf.gradients(
                        self._discriminator_loss_logits, self._sample_input
                    )[0]
                ) - 1
            ) * configs['wgan_gradient_penalty']
            
            self._discriminator_loss = self._discriminator_loss_logits + self._discriminator_loss_gradient
    
            self._generator_loss = tf.reduce_mean(
                -self._generator_output
            )
            
        elif configs['gan_type'] == 'lsgan':
            self._generator_output = self.generator_discriminator.output
            self._sample_output = self.sample_discriminator.output
            
            self._discriminator_loss = tf.reduce_mean(
                tf.square(2 * self._label - 1 - self._sample_output)
            )
            
            self._generator_loss = tf.reduce_mean(
                tf.square(self._generator_output - 1)
            )
        
        elif configs['gan_type'] == 'original':
            self._generator_output = tf.sigmoid(self.generator_discriminator.output)
            self._sample_output = tf.sigmoid(self.sample_discriminator.output)
    
            self._discriminator_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self._label, logits=self.sample_discriminator.output
                )
            )
    
            self._generator_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(self.generator_discriminator.output),
                    logits=self.generator_discriminator.output
                )
            )
        else:
            raise ValueError('Unsupported GAN type!')

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



class DiscriminatorNet(object):
    
    def __init__(self, input_tensor, hidden_layers, output_size, is_training, configs, reuse=False):
        out = input_tensor

        for i, size in enumerate(hidden_layers):
            out = tf.layers.dense(
                out, size,
                name='fc_{}'.format(i), reuse=reuse,
                kernel_initializer=configs['discriminator_weight_initializer'],
            )
            
            if configs['discriminator_hidden_activation'] == 'relu':
                out = tf.nn.relu(out)
            elif configs['discriminator_hidden_activation'] == 'leaky_relu':
                out = tf.maximum(0.1 * out, out)
            else:
                raise ValueError('Unsupported activation type')
            
            if configs['batch_normalize_discriminator']:
                out = tf.layers.batch_normalization(
                    out, name='bn_{}'.format(i),
                    training=is_training, reuse=reuse
                )

        self._output = tf.layers.dense(
            out, output_size,
            name='fc_out'.format(i),
            kernel_initializer=configs['discriminator_weight_initializer'],
            reuse=reuse
        )
        
    @property
    def output(self):
        return self._output