import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import io

import numpy as np
import scipy
import tensorflow as tf
import tflearn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

from rllab.misc import logger

from sandbox.young_clgan.gan.gan import FCGAN
from sandbox.young_clgan.logging import HTMLReport, format_dict
from rllab.misc.instrument import run_experiment_lite, VariantGenerator


def plot_array(array):
    file = io.BytesIO()
    plt.imshow(array, cmap='gray')
    plt.savefig(file, format='png')
    file.seek(0)
    plt.close()
    return scipy.misc.imread(file)
    

def run_task(variant):
    
    gan_configs = {
        'batch_size': 512,
        'generator_output_activation': 'sigmoid',
        'generator_optimizer': tf.train.AdamOptimizer(variant['generator_learning_rate']),
        'discriminator_optimizer': tf.train.AdamOptimizer(variant['discriminator_learning_rate']),
        'reset_generator_optimizer': False,
        'reset_discriminator_optimizer': False,
    }
    
    if variant['generator_init'] == 'xavier':
        gan_configs['generator_weight_initializer'] = tf.contrib.layers.xavier_initializer()
    else:
        gan_configs['generator_weight_initializer'] = tflearn.initializations.truncated_normal(stddev=variant['generator_init'])
    
    gan = FCGAN(
        generator_output_size=28 * 28,
        discriminator_output_size=1,
        generator_layers=[1200, 1200],
        discriminator_layers=[500, 500],
        noise_size=100,
        tf_session=tf.Session(),
        configs=gan_configs,
    )
    
    log_dir = logger.get_snapshot_dir()
    report = HTMLReport(os.path.join(log_dir, 'report.html'), images_per_row=5)
    report.add_header('MNIST GAN')
    report.add_text(format_dict(variant))
    report.save()
    
    mnist = input_data.read_data_sets(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), 'MNIST_data'),
        one_hot=True
    )
    
    # mnist_data_normalized = 2 * (mnist.train.images - 0.5)
    mnist_data_normalized = mnist.train.images
    
    
    for outer_iter in range(10):
        dloss, gloss = gan.train(
            mnist_data_normalized, np.ones((mnist_data_normalized.shape[0], 1)),
            outer_iters=variant['outer_iters'], generator_iters=variant['generator_iters'],
            discriminator_iters=variant['discriminator_iters']
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
    vg.add('generator_init', ['xavier'])
    vg.add('generator_iters', [2])
    vg.add('discriminator_iters', [1])
    vg.add('generator_learning_rate', [0.001, 0.01, 0.1])
    vg.add('discriminator_learning_rate', [0.001, 0.01, 0.1])
    vg.add('outer_iters', [1000])
    
    
    for variant in vg.variants(randomized=False):
        run_experiment_lite(
            stub_method_call=run_task,
            mode='local',
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            seed=int(time.time()),
            exp_prefix='mnist_gan',
            variant=variant,
            # exp_name=exp_name,
        )