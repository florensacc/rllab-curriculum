import os
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''

from sandbox.young_clgan.lib.utils import initialize_parallel_sampler
initialize_parallel_sampler()

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

from sandbox.young_clgan.lib.envs.base import UniformGoalGenerator, FixedGoalGenerator, update_env_goal_generator
from sandbox.young_clgan.lib.envs.point_env import PointEnv
from sandbox.young_clgan.lib.goal import *
from sandbox.young_clgan.lib.logging import *


EXPERIMENT_TYPE = 'trpo_baseline'


if __name__ == '__main__':
    log_config = format_experiment_log_path(__file__, EXPERIMENT_TYPE)

    make_log_dirs(log_config)

    hyperparams = AttrDict(
        goal_range=15,
        horizon=200,
        outer_iters=50,
        inner_iters=50,
        pg_batch_size=20000,
    )

    report = HTMLReport(log_config.report_file)

    report.add_header("{}, {}".format(EXPERIMENT_TYPE, log_config.experiment_date_host))
    report.add_text(format_dict(hyperparams))

    env = normalize(PointEnv(
        FixedGoalGenerator([0.1, 0.1])
    ))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 8 hidden units.
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    for outer_iter in range(hyperparams.outer_iters):
        with ExperimentLogger(log_config.log_dir, outer_iter):
            update_env_goal_generator(
                env,
                UniformGoalGenerator(
                    np.random.uniform(
                        -hyperparams.goal_range, hyperparams.goal_range,
                        size=(1000, 2)
                    ).tolist()
                )
            )

            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=hyperparams.pg_batch_size,
                max_path_length=hyperparams.horizon,
                n_itr=hyperparams.inner_iters,
                discount=0.995,
                step_size=0.01,
                # Uncomment both lines (this and the plot parameter below) to enable plotting
                plot=False,
            )
            algo.train()

            img = plot_policy_reward(
                policy, env, hyperparams.goal_range,
                horizon=hyperparams.horizon,
                fname='{}/policy_reward_{}.png'.format(log_config.plot_dir, outer_iter),
            )
            report.add_image(img, 'policy performance\n itr: {}'.format(outer_iter))
            report.save()
