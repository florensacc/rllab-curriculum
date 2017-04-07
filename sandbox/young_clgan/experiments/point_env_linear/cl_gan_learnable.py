import os
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''

import multiprocessing

# Symbols that need to be stubbed
from rllab.misc.instrument import stub, run_experiment_lite

import time
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')

from sandbox.young_clgan.logging import *

EXPERIMENT_TYPE = 'cl_gan_learnable'

from sandbox.young_clgan.experiments.point_env_linear.cl_gan_learnable_algo import CLGANPointEnvLinear

stub(globals())

from sandbox.young_clgan.utils import AttrDict


if __name__ == '__main__':

    log_config = format_experiment_log_path(__file__, EXPERIMENT_TYPE)
    make_log_dirs(log_config)

    random.seed(int(time.time()))
    np.random.seed(int(time.time()))

    report = HTMLReport(log_config.report_file)

    hyperparams = AttrDict(
        horizon=200,
        goal_size=2,
        goal_range=15,
        goal_noise_level=1,
        min_reward=5,
        max_reward=3000,
        improvement_threshold=10,
        outer_iters=50,
        inner_iters=50,
        pg_batch_size=20000,
        gan_outer_iters=5,
        gan_discriminator_iters=200,
        gan_generator_iters=5,
        gan_noise_size=4,
        gan_generator_layers=[256, 256],
        gan_discriminator_layers=[128, 128],
        experiment_type=EXPERIMENT_TYPE,
    )

    algo = CLGANPointEnvLinear(hyperparams)

    run_experiment_lite(
        algo.train(),
        n_parallel=multiprocessing.cpu_count(),
        use_cloudpickle=False,
        snapshot_mode="none",
        use_gpu=False,
    )
