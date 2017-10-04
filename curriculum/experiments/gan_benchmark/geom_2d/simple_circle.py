import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import io

import numpy as np
import scipy
import tensorflow as tf
import tflearn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from rllab.misc import logger

from curriculum.gan.gan import FCGAN
from curriculum.logging import HTMLReport, format_dict
from rllab.misc.instrument import run_experiment_lite, VariantGenerator


def plot_samples(samples):
    file = io.BytesIO()
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
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
    plt.savefig(file, format='png')
    file.seek(0)
    plt.close()
    return scipy.misc.imread(file)


def run_task(variant):
    
    gan_configs = {
        'batch_size': 64,
        'generator_output_activation': 'tanh',
        'generator_optimizer': tf.train.RMSPropOptimizer(variant['generator_learning_rate']),
        'discriminator_optimizer': tf.train.RMSPropOptimizer(variant['discriminator_learning_rate']),
        'batch_normalize_discriminator': False,
        'batch_normalize_generator': False,
        'gan_type': 'lsgan',
    }
    
    if variant['generator_init'] == 'xavier':
        gan_configs['generator_weight_initializer'] = tf.contrib.layers.xavier_initializer()
    else:
        gan_configs['generator_weight_initializer'] = tflearn.initializations.truncated_normal(stddev=variant['generator_init'])
    
    gan = FCGAN(
        generator_output_size=2,
        discriminator_output_size=1,
        generator_layers=[200, 200],
        discriminator_layers=[128, 128],
        noise_size=5,
        tf_session=tf.Session(),
        configs=gan_configs,
    )
    
    log_dir = logger.get_snapshot_dir()
    report = HTMLReport(
        os.path.join(log_dir, 'report.html'), images_per_row=2,
        default_image_width=500
    )
    report.add_header('Simple Circle Sampling')
    report.add_text(format_dict(variant))
    report.save()
    
    rand_theta = np.random.uniform(0, 2 * np.pi, size=(5000, 1))
    data = np.hstack([0.5 * np.cos(rand_theta), 0.5 * np.sin(rand_theta)])
    data = data + np.random.normal(scale=0.05, size=data.shape)
    
    report.add_image(
        plot_samples(data[:500, :]), 'Real data'
    )
    
    
    # for outer_iter in range(30):
    #     loss = gan.train_discriminator(data, data[:, 0:1] < 0, 100)
    #     logger.log(str(loss))
        
    # report.add_image(
    #     plot_dicriminator(gan)
    # )
    # report.save()
    
    # logger.log('Now training generator')
        
    # for outer_iter in range(30):
    #     loss = gan.train_generator(np.random.randn(1000, 2), 100)
    #     logger.log(str(loss))
        
    # generated_samples, _ = gan.sample_generator(50)
    # report.add_image(
    #     plot_samples(generated_samples)
    # )
    
    for outer_iter in range(30):
        dloss, gloss = gan.train(
            data, np.ones((data.shape[0], 1)),
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
        generated_samples, _ = gan.sample_generator(50)
        report.add_image(
            plot_samples(generated_samples)
        )
        report.add_image(
            plot_dicriminator(gan)
        )
        
        report.save()
        
    
if __name__ == '__main__':
    vg = VariantGenerator()
    vg.add('generator_init', ['xavier'])
    vg.add('generator_iters', [1])
    vg.add('discriminator_iters', [1])
    vg.add('generator_learning_rate', [0.001])
    vg.add('discriminator_learning_rate', [0.001])
    vg.add('outer_iters', [500])
    
    
    for variant in vg.variants(randomized=False):
        run_experiment_lite(
            stub_method_call=run_task,
            mode='local',
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            seed=int(time.time()),
            exp_prefix='simple_circle_gan',
            variant=variant,
            # exp_name=exp_name,
            print_command=False
        )