import os

from sandbox.young_clgan.experiments.point_maze.trpo_baseline_algo import TRPOPointEnvMaze

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''

import multiprocessing

# Symbols that need to be stubbed
from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.sampler.stateful_pool import singleton_pool

import random

import matplotlib
matplotlib.use('Agg')

from sandbox.young_clgan.logging import *


EXPERIMENT_TYPE = 'trpo'

use_stub = True

use_ec2 = True

if use_ec2:
    use_stub = True
    mode = "ec2"
    n_parallel = 4
else:
    mode = "local"
    n_parallel = multiprocessing.cpu_count()
    #n_parallel = 1

if use_ec2:
    seeds = [1, 11, 21, 31, 41]
else:
    seeds = [1]

#from sandbox.young_clgan.experiments.point_linear.cl_gan_algo import CLGANPointEnvLinear

if use_stub:
    stub(globals())

from sandbox.young_clgan.utils import AttrDict

if __name__ == '__main__':



    hyperparams = AttrDict(
        horizon=400,
        goal_size=2,
        goal_range=10,
        goal_noise_level=1,
        min_reward=5,
        max_reward=6000,
        improvement_threshold=10,
        outer_iters=200,
        inner_iters=5,
        pg_batch_size=20000,
        discount=0.998,
        gae_lambda=0.995,
        num_new_goals=200,
        num_old_goals=200,
        experiment_type=EXPERIMENT_TYPE,
        # Unused - kept just so that viskit will not have issues with the comparison
        gan_outer_iters=5,
        gan_discriminator_iters=200,
        gan_generator_iters=5,
        gan_noise_size=4,
        gan_generator_layers=[256, 256],
        gan_discriminator_layers=[128, 128],
    )

    algo = TRPOPointEnvMaze(hyperparams)

    if use_stub:
        for seed in seeds:
            run_experiment_lite(
                algo.train(),
                pre_commands=['export MPLBACKEND=Agg',
                              'pip install --upgrade pip',
                              'pip install --upgrade -I tensorflow',
                              'pip install git+https://github.com/tflearn/tflearn.git',
                              'pip install dominate',
                              'pip install scikit-image',
                              'conda install numpy -n rllab3 -y',
                              ],
                n_parallel=n_parallel,
                use_cloudpickle=False,
                snapshot_mode="none",
                use_gpu=False,
                mode=mode,
                sync_s3_html=True,
                exp_prefix='trpo-maze5',
                seed=seed
            )
    else:
        algo.train()

