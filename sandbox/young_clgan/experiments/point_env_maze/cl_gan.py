import os

from sandbox.young_clgan.experiments.point_env_maze.cl_gan_algo import CLGANPointEnvMaze

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''

import multiprocessing

# Symbols that need to be stubbed
import rllab
from rllab.algos.base import RLAlgorithm
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
import rllab.misc.logger
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sampler.stateful_pool import singleton_pool

import time
import datetime
import random

import numpy as np
import scipy
import tensorflow as tf
import tflearn
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sandbox.young_clgan.lib.envs.base import UniformListGoalGenerator, FixedGoalGenerator, update_env_goal_generator
from sandbox.young_clgan.lib.envs.point_env import PointEnv
from sandbox.young_clgan.lib.goal import *
from sandbox.young_clgan.lib.logging import *


EXPERIMENT_TYPE = 'cl_gan'

use_stub = False

use_ec2 = True

if use_ec2:
    use_stub = True
    mode = "ec2"
    n_parallel = 4
else:
    mode = "local"
    n_parallel = multiprocessing.cpu_count()

#from sandbox.young_clgan.experiments.point_env_linear.cl_gan_algo import CLGANPointEnvLinear

if use_stub:
    stub(globals())

from sandbox.young_clgan.lib.utils import AttrDict

if __name__ == '__main__':



    hyperparams = AttrDict(
        horizon=200,
        goal_size=2,
        goal_range=15,
        goal_noise_level=1,
        min_reward=5,
        max_reward=1000,
        improvement_threshold=10,
        outer_iters=200,
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

    algo = CLGANPointEnvMaze(hyperparams)

    if use_stub:
        run_experiment_lite(
            algo.train(),
            n_parallel=n_parallel,
            use_cloudpickle=False,
            snapshot_mode="none",
            use_gpu=False,
            mode=mode,
        )
    else:
        algo.train()
