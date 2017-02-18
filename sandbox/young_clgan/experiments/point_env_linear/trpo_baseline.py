import os
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''

import multiprocessing

import os.path
import datetime
import multiprocessing
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import joblib

import rllab
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import rllab.misc.logger

from sandbox.young_clgan.lib.envs.base import UniformListGoalGenerator, FixedGoalGenerator, update_env_goal_generator
from sandbox.young_clgan.lib.envs.point_env import PointEnv
from sandbox.young_clgan.lib.goal import *
from sandbox.young_clgan.lib.logging import *


EXPERIMENT_TYPE = 'trpo_baseline'

from sandbox.young_clgan.experiments.point_env_linear.trpo_baseline_algo import TRPOPointEnvLinear

stub(globals())

from sandbox.young_clgan.lib.utils import AttrDict


if __name__ == '__main__':

    hyperparams = AttrDict(
        horizon=200,
        goal_size=2,
        goal_range=15,
        max_reward=6000,
        outer_iters=200,
        inner_iters=50,
        pg_batch_size=20000,
        experiment_type=EXPERIMENT_TYPE,
    )

    algo = TRPOPointEnvLinear(hyperparams)

    run_experiment_lite(
        algo.train(),
        n_parallel=multiprocessing.cpu_count(),
        use_cloudpickle=False,
        snapshot_mode="none",
        use_gpu=False,
    )
