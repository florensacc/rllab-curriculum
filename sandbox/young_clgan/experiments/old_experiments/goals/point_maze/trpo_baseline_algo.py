import os

from sandbox.young_clgan.envs.maze.maze_evaluate import test_and_plot_policy
from sandbox.young_clgan.lib.envs.maze.point_maze_env import PointMazeEnv
from sandbox.young_clgan.lib.logging import HTMLReport
from sandbox.young_clgan.lib.logging import format_dict
from sandbox.young_clgan.lib.logging.visualization import save_image, plot_line_graph

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''

# Symbols that need to be stubbed
from rllab.algos.base import RLAlgorithm
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

import time
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')

from sandbox.young_clgan.lib.envs.base import UniformListStateGenerator, FixedStateGenerator, update_env_state_generator
#from sandbox.young_clgan.lib.logging import *
#from sandbox.young_clgan.lib.logging.logger import ExperimentLogger

from sandbox.young_clgan.logging.logger import ExperimentLogger, AttrDict, format_experiment_log_path, make_log_dirs
from rllab.misc import logger

EXPERIMENT_TYPE = 'trpo_maze'


class TRPOPointEnvMaze(RLAlgorithm):

    def __init__(self, hyperparams):
        self.hyperparams = AttrDict(hyperparams)

    def train(self):
        hyperparams = self.hyperparams
        log_config = format_experiment_log_path(
            __file__, EXPERIMENT_TYPE
        )
        make_log_dirs(log_config)

        random.seed(int(time.time()))
        np.random.seed(int(time.time()))

        report = HTMLReport(log_config.report_file)
        # logger.set_snapshot_dir(log_config["log_dir"])
        # logger.set_tf_summary_dir(log_config["log_dir"])
        # print("Logger snapshot dir: ", logger.get_snapshot_dir())

        report.add_header("{}, {}".format(EXPERIMENT_TYPE, log_config.experiment_date_host))
        report.add_text(format_dict(hyperparams))

        env = normalize(PointMazeEnv(
            goal_generator=FixedStateGenerator([0.1, 0.1])
        ))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            # Fix the variance since different goals will require different variances, making this parameter hard to learn.
            learn_std=False
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        rewards, avg_success, heatmap = test_and_plot_policy(policy, env)
        # reward_img = save_image(fname='{}/policy_reward_init.png'.format(log_config.plot_dir))
        report.add_image(
            heatmap,
            'policy performance initialization\n'
        )

        report.save()
        report.new_row()

        all_mean_rewards = []
        all_coverage = []
        

        for outer_iter in range(hyperparams.outer_iters):
            with ExperimentLogger(log_config.log_dir, outer_iter, hold_outter_log=True):
                print("Sampling goals uniformly at random")
                update_env_state_generator(
                    env,
                    UniformListStateGenerator(
                        np.random.uniform(
                            -hyperparams.goal_range, hyperparams.goal_range,
                            size=(1000, hyperparams.goal_size)
                        ).tolist()
                    )
                )

                print("Training the algorithm")
                algo = TRPO(
                    env=env,
                    policy=policy,
                    baseline=baseline,
                    batch_size=hyperparams.pg_batch_size,
                    max_path_length=hyperparams.horizon,
                    n_itr=hyperparams.inner_iters,
                    discount=0.9975,
                    step_size=0.01,
                    plot=False,
                )

                algo.train()

            rewards, avg_success, heatmap = test_and_plot_policy(policy, env)
            reward_img = save_image(fname='{}/policy_reward_{}.png'.format(log_config.plot_dir, outer_iter))

            mean_rewards = np.mean(rewards)
            coverage = np.mean(rewards >= hyperparams.max_reward)

            all_mean_rewards.append(mean_rewards)
            all_coverage.append(coverage)

            # logger.set_snapshot_dir(log_config["log_dir"])
            # logger.set_tf_summary_dir(log_config["log_dir"])
            print("Logger snapshot dir: ", logger.get_snapshot_dir())
            with logger.tabular_prefix('Outer_'):
                logger.record_tabular('MeanRewards', mean_rewards)
                logger.record_tabular('Coverage', coverage)
            logger.dump_tabular(with_prefix=False)

            report.add_image(
                reward_img,
                'policy performance\n itr: {} \nmean_rewards: {} \ncoverage: {}'.format(
                    outer_iter, all_mean_rewards[-1],
                    all_coverage[-1]
                )
            )

            report.save()

            print("Adding a new row to the report")
            report.new_row()
                
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
        
