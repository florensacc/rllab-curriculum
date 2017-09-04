import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import io

import numpy as np
import scipy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tflearn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rllab.misc import logger

from sandbox.ignasi.gan.gan import FCGAN, batch_feed_array
from sandbox.ignasi.logging import HTMLReport, format_dict
from rllab.misc.instrument import run_experiment_lite, VariantGenerator


def real_valued_nn(input_tensor, n_out, hidden_layers, name_prefix,
                   reuse=False, activation=tf.nn.relu):
    out = input_tensor
    for i, n_units in enumerate(hidden_layers):
        out = tf.layers.dense(
            out, n_units, name='{}_dense_{}'.format(name_prefix, i), reuse=reuse,
            # kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
        )
        out = activation(
            out, name='{}_activation_{}'.format(name_prefix, i)
        )

    out = tf.layers.dense(
        out, n_out, name='{}_dense_out'.format(name_prefix), reuse=reuse,
        # kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
    )
    return out


def generator(noise_input, n_out=28*28, reuse=False):
    return real_valued_nn(
        noise_input, n_out, [1500, 1500], 'generator', reuse=reuse,
        activation=tf.nn.relu
    )
    
    
def discriminator(input_tensor, reuse=False):
    return real_valued_nn(
        input_tensor, 1, [500, 500], 'discriminator', reuse=reuse,
        activation=lambda x, name: tf.maximum(x, 0.2 * x, name=name)
    )


class SimpleGAN(object):
    
    def __init__(self, noise_size, tf_session):
        
        self.tf_session = tf_session
        self.noise_size = noise_size
        self.batch_size = 64
        
        self.noise_placeholder = tf.placeholder(
            dtype=tf.float32, shape=[None, noise_size]
        )
        
        self.sample_placeholder = tf.placeholder(
            dtype=tf.float32, shape=[None, 784]
        )
        
        with tf.variable_scope('generator'):
            self.generator_output = tf.sigmoid(
                generator(self.noise_placeholder)
            )
        
        with tf.variable_scope('discriminator'):
            self.sample_discriminator_output = discriminator(self.sample_placeholder)
            
            self.generator_discriminator_output = discriminator(self.generator_output, reuse=True)

            
        self.generator_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            'generator'
        )
        self.discriminator_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            'discriminator'
        )
        
        # self.discriminator_loss_1 = -tf.reduce_mean(
        #     self.sample_discriminator_output
        # )
        # self.discriminator_loss_1 += tf.nn.relu(
        #     tf.nn.l2_loss(
        #         tf.gradients(
        #             self.discriminator_loss_1, self.sample_placeholder
        #         )[0]
        #     ) - 1
        # ) * 10000
        
        # self.discriminator_loss_2 = tf.reduce_mean(
        #     self.generator_discriminator_output
        # )
        # self.discriminator_loss_2 += tf.nn.relu(
        #     tf.nn.l2_loss(
        #         tf.gradients(
        #             self.discriminator_loss_2, self.generator_output
        #         )[0]
        #     ) - 1
        # )  * 10000
        
        self.discriminator_loss_1 = tf.reduce_mean(
            tf.square(self.sample_discriminator_output - 1)
        )
        self.discriminator_loss_2 = tf.reduce_mean(
            tf.square(self.generator_discriminator_output + 1)
        )
        
        
        self.discriminator_loss = (
            self.discriminator_loss_1 + self.discriminator_loss_2
        )
        
        # self.generator_loss = -tf.reduce_mean(
        #     self.generator_discriminator_output
        # )
        
        self.generator_loss = tf.reduce_mean(
            tf.square(self.generator_discriminator_output - 1)
        )
        
        self.generator_train_op = tf.train.RMSPropOptimizer(0.001).minimize(
            self.generator_loss,
            var_list=self.generator_variables
        )
        
        self.discriminator_train_op = tf.train.RMSPropOptimizer(0.001).minimize(
            self.discriminator_loss,
            var_list=self.discriminator_variables
        )
        
        self.init_op = tf.global_variables_initializer()
        
        self.tf_session.run(self.init_op)
            
        
    def sample_random_noise(self, size):
        return np.random.randn(size, self.noise_size)
        
    def sample_generator(self, size):
        noise = self.sample_random_noise(size)
        return self.tf_session.run(
            self.generator_output,
            {self.noise_placeholder: noise}
        ), noise
        
    def discriminator_predict(self, X):
        return self.tf_session.run(
            self.sample_discriminator_output,
            {self.sample_placeholder: X}
        )
        
    def train(self, X, outer_iters):
        batch_X = batch_feed_array(X, self.batch_size)
        
        
        for i in range(outer_iters):
            noise = self.sample_random_noise(self.batch_size)
            train_X = next(batch_X)
    
            
            dloss, _ = self.tf_session.run(
                [self.discriminator_loss, self.discriminator_train_op],
                {self.sample_placeholder: train_X,
                 self.noise_placeholder: noise}
            )
            
            gloss, _ = self.tf_session.run(
                [self.generator_loss, self.generator_train_op],
                {self.noise_placeholder: noise}
            )
            
            gloss, _ = self.tf_session.run(
                [self.generator_loss, self.generator_train_op],
                {self.noise_placeholder: noise}
            )
        
        return dloss, gloss
        



def plot_samples(samples):
    file = io.BytesIO()
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.savefig(file, format='png')
    file.seek(0)
    plt.close()
    return scipy.misc.imread(file)
    
    
def plot_dicriminator(gan, grid_size=60):
    x, y = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))
    grid_shape = x.shape
    samples = np.hstack([
        x.flatten().reshape(-1, 1),
        y.flatten().reshape(-1, 1)
    ])
    z = gan.discriminator_predict(samples)
    z = z.reshape(grid_shape)
    file = io.BytesIO()
    plt.figure()
    plt.clf()
    plt.pcolormesh(x, y, z, cmap='jet')
    plt.colorbar()
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.savefig(file, format='png')
    file.seek(0)
    plt.close()
    return scipy.misc.imread(file)


def run_task(variant):
    
    
    
    log_dir = logger.get_snapshot_dir()
    report = HTMLReport(
        os.path.join(log_dir, 'report.html'), images_per_row=2,
        default_image_width=500
    )
    report.add_header('Debugging MNIST GAN')
    report.add_text(format_dict(variant))
    report.save()
    
    gan = SimpleGAN(noise_size=100, tf_session=tf.Session())
    
    mnist = input_data.read_data_sets(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), 'MNIST_data'),
        one_hot=True
    )
    
    mnist_data_normalized = mnist.train.images
    
    
    for outer_iter in range(10):
        dloss, gloss = gan.train(
            mnist_data_normalized,
            outer_iters=variant['outer_iters']
        )
        logger.log(
            'Outer iteration: {}, disc loss: {}, gen loss: {}'.format(
                outer_iter, dloss, gloss
            )
        )
        report.add_text(
            'Outer iteration: {}, disc loss: {}, gen loss: {}'.format(
                outer_iter, dloss, gloss
            )
        )
        sampled_images, _ = gan.sample_generator(10)
        for arr in sampled_images:
            report.add_image(
                arr.reshape(28, 28), width=150
            )
        
        report.save()
        
    
if __name__ == '__main__':
    vg = VariantGenerator()
    vg.add('outer_iters', [500])
    
    
    for variant in vg.variants(randomized=False):
        run_experiment_lite(
            stub_method_call=run_task,
            mode='local',
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            seed=int(time.time()),
            exp_prefix='debug_mnist_gan',
            variant=variant,
            # exp_name=exp_name,
        )