import os
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''

import time
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import rllab
from rllab.algos.base import RLAlgorithm
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import rllab.misc.logger

from sandbox.young_clgan.lib.envs.base import UniformListGoalGenerator, FixedGoalGenerator, update_env_goal_generator
from sandbox.young_clgan.envs.point_env import PointEnv
from sandbox.young_clgan.goal import *
from sandbox.young_clgan.logging import *


EXPERIMENT_TYPE = 'trpo_baseline'


class TRPOPointEnvLinear(RLAlgorithm):
    
    def __init__(self, hyperparams):
        self.hyperparams = AttrDict(hyperparams)
        
    def train(self):
        hyperparams = self.hyperparams
        
        log_config = format_experiment_log_path(__file__, EXPERIMENT_TYPE)
    
        log_config = format_experiment_log_path(
            __file__, EXPERIMENT_TYPE
        )
        make_log_dirs(log_config)
        

        random.seed(int(time.time()))
        np.random.seed(int(time.time()))
    
    
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
        
        all_mean_rewards = []
        all_coverage = []
    
        for outer_iter in range(hyperparams.outer_iters):
            with ExperimentLogger(log_config.log_dir, outer_iter):
                update_env_goal_generator(
                    env,
                    UniformListGoalGenerator(
                        np.random.uniform(
                            -hyperparams.goal_range, hyperparams.goal_range,
                            size=(1000, hyperparams.goal_size)
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
    
                img, rewards = plot_policy_reward(
                    policy, env, hyperparams.goal_range,
                    horizon=hyperparams.horizon,
                    fname='{}/policy_reward_{}.png'.format(log_config.plot_dir, outer_iter),
                    return_rewards=True,
                )
                
                all_mean_rewards.append(np.mean(rewards))
                all_coverage.append(np.mean(rewards >= hyperparams.max_reward))
                
                report.add_image(
                    img,
                    'policy performance\n itr: {} \nmean_rewards: {} \ncoverage: {}'.format(
                        outer_iter, all_mean_rewards[-1],
                        all_coverage[-1]
                    )
                )
                report.save()
                
        img = plot_line_graph(
            '{}/mean_rewards.png'.format(log_config.plot_dir),
            range(hyperparams.outer_iters), all_mean_rewards
        )
        report.add_image(img, 'Mean rewards', width=500)
        
        img = plot_line_graph(
            '{}/coverages.png'.format(log_config.plot_dir),
            range(hyperparams.outer_iters), all_coverage
        )
        report.add_image(img, 'Coverages', width=500)
        report.save()
