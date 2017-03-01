import os
import os.path as osp
import datetime

from sandbox.young_clgan.experiments.point_env_maze.maze_evaluate import test_and_plot_policy
from sandbox.young_clgan.lib.envs.maze.point_maze_env import PointMazeEnv
from sandbox.young_clgan.lib.goal.utils import GoalCollection
from sandbox.young_clgan.lib.logging import HTMLReport
from sandbox.young_clgan.lib.logging import format_dict
from sandbox.young_clgan.lib.logging.visualization import save_image, plot_gan_samples, plot_labeled_samples, \
    plot_line_graph

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
import tensorflow as tf
import tflearn
import matplotlib
matplotlib.use('Agg')

from sandbox.young_clgan.lib.envs.base import UniformListGoalGenerator, FixedGoalGenerator, update_env_goal_generator, generate_initial_goals
from sandbox.young_clgan.lib.goal import *
#from sandbox.young_clgan.lib.logging import *
#from sandbox.young_clgan.lib.logging.logger import ExperimentLogger

from sandbox.young_clgan.lib.logging.logger import ExperimentLogger, AttrDict, format_experiment_log_path, make_log_dirs
from rllab.misc import logger



EXPERIMENT_TYPE = 'cl_gan_maze'


class CLGANPointEnvMaze(RLAlgorithm):

    def __init__(self, hyperparams):
        self.hyperparams = AttrDict(hyperparams)
        
    def convert_label(self, labels):
        # Put good goals last so they will be plotted on top of other goals and be most visible.
        classes = {
            0: 'Other',
            1: 'Low rewards',
            2: 'High rewards',
            3: 'Unlearnable',
            4: 'Good goals',
        }
        new_labels = np.zeros(labels.shape[0], dtype=int)
        new_labels[np.logical_and(labels[:, 0], labels[:, 1])] = 4
        new_labels[labels[:, 0] == False] = 1
        new_labels[labels[:, 1] == False] = 2
        new_labels[
            np.logical_and(
                np.logical_and(labels[:, 0], labels[:, 1]),
                labels[:, 2] == False
            )
        ] = 3
    
        return new_labels, classes

    def train(self):
        gan_configs = {
            'batch_size': 128,
            'generator_activation': 'relu',
            'discriminator_activation': 'relu',
            'generator_optimizer': tf.train.AdamOptimizer(0.001),
            'discriminator_optimizer': tf.train.AdamOptimizer(0.001),
            'generator_weight_initializer': tflearn.initializations.truncated_normal(stddev=0.05),
            'discriminator_weight_initializer': tflearn.initializations.truncated_normal(),
            'discriminator_batch_noise_stddev': 1e-2,
        }


        hyperparams = self.hyperparams
        experiment_date_host = datetime.datetime.today().strftime(
            "%Y-%m-%d_%H-%M-%S_{}".format(os.uname()[1])
        )

        # log_config = format_experiment_log_path(
        #     __file__, EXPERIMENT_TYPE
        # )
        # make_log_dirs(log_config)

        random.seed(int(time.time()))
        np.random.seed(int(time.time()))

        # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
        logger.log("Initializing report and plot_policy_reward...")
        log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
        report = HTMLReport(osp.join(log_dir, 'report.html'))

        report.add_header("{}, {}".format(EXPERIMENT_TYPE, experiment_date_host))
        report.add_text(format_dict(hyperparams))

        tf_session = tf.Session()

        gan = GoalGAN(
            goal_size=hyperparams.goal_size,
            evaluater_size=3,
            goal_range=hyperparams.goal_range,
            goal_noise_level=hyperparams.goal_noise_level,
            generator_layers=hyperparams.gan_generator_layers,
            discriminator_layers=hyperparams.gan_discriminator_layers,
            noise_size=hyperparams.gan_noise_size,
            tf_session=tf_session,
            configs=gan_configs
        )

        reward_dist_threshold = 0.3

        env = normalize(PointMazeEnv(
            goal_generator=FixedGoalGenerator([0.1, 0.1]),
            reward_dist_threshold=reward_dist_threshold
        ))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            # Fix the variance since different goals will require different variances, making this parameter hard to learn.
            learn_std=False
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        test_and_plot_policy(policy, env)
        reward_img = save_image(fname=osp.join(log_dir, 'policy_reward_init.png'))
        report.add_image(
            reward_img,
            'policy performance initialization\n'
        )

        # logger.log("Pretraining the gan for uniform sampling")
        # gan.pretrain_uniform()

        logger.log("Pretraining the gan with initial goals")
        gan.pretrain(
            generate_initial_goals(env, policy, hyperparams.goal_range)
        )

        logger.log("Plotting GAN samples")
        img = plot_gan_samples(gan, hyperparams.goal_range, osp.join(log_dir, 'start.png'))
        report.add_image(img, 'GAN initialization')

        report.save()
        report.new_row()

        all_goals = GoalCollection(distance_threshold=reward_dist_threshold)
        #all_goals = GoalCollection(distance_threshold=0)
        
        all_mean_rewards = []
        all_coverage = []
        

        for outer_iter in range(hyperparams.outer_iters):

            # Train GAN
            logger.log("Sampling goals from the GAN")
            raw_goals, _ = gan.sample_goals_with_noise(hyperparams.num_new_goals)

            if outer_iter > 0:
                old_goals = all_goals.sample(hyperparams.num_old_goals)
                goals = np.vstack([raw_goals, old_goals])
            else:
                goals = raw_goals

            #all_goals.append(raw_goals)

            logger.log("Evaluating goals before training")
            rewards_before = evaluate_goals(goals, env, policy, hyperparams.horizon, n_processes=12)

            with ExperimentLogger(log_dir, outer_iter, hold_outter_log=True):
                logger.log("Updating the environment goal generator")
                update_env_goal_generator(
                    env,
                    UniformListGoalGenerator(
                        goals.tolist()
                    )
                )

                logger.log("Training the algorithm")
                algo = TRPO(
                    env=env,
                    policy=policy,
                    baseline=baseline,
                    batch_size=hyperparams.pg_batch_size,
                    max_path_length=hyperparams.horizon,
                    n_itr=hyperparams.inner_iters,
                    discount=hyperparams.discount,
                    step_size=0.01,
                    plot=False,
                    gae_lambda=hyperparams.gae_lambda
                )

                algo.train()

            rewards = test_and_plot_policy(policy, env)
            reward_img = save_image(fname=osp.join(log_dir,'policy_reward_{}.png'.format(outer_iter)))

            mean_rewards = np.mean(rewards)
            coverage = np.mean(rewards >= hyperparams.max_reward)

            all_mean_rewards.append(mean_rewards)
            all_coverage.append(coverage)

            logger.log("Logger snapshot dir: ", logger.get_snapshot_dir())
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

            logger.log("Labeling the goals")
            labels = label_goals(
                goals, env, policy, hyperparams.horizon,
                min_reward=hyperparams.min_reward,
                max_reward=hyperparams.max_reward,
                old_rewards=rewards_before,
                improvement_threshold=hyperparams.improvement_threshold
            )

            logger.log("Training the GAN")
            gan.train(
                goals, labels,
                hyperparams.gan_outer_iters,
                hyperparams.gan_generator_iters,
                hyperparams.gan_discriminator_iters,
                suppress_generated_goals=True
            )

            logger.log("Converting the labels")
            plot_labels, classes = self.convert_label(labels)

            logger.log("Plotting the labeled samples")
            img = plot_labeled_samples(
                samples=goals, sample_classes=plot_labels,
                text_labels=classes, limit=hyperparams.goal_range + 5,
                fname=osp.join(log_dir, 'sampled_goals_{}.png'.format(outer_iter)),
            )

            logger.log("Adding image to the report")
            report.add_image(img, 'goals\n itr: {}'.format(outer_iter), width=500)

            logger.log("Saving the report")
            report.save()

            logger.log("Adding a new row to the report")
            report.new_row()

            # append new goals to list of all goals (replay buffer): Not the low reward ones!!
            filtered_raw_goals = [goal for goal, label in zip(goals, labels) if label[0] == 1]
            all_goals.append(filtered_raw_goals)
                
        img = plot_line_graph(
            osp.join(log_dir, 'mean_rewards.png'),
            range(hyperparams.outer_iters), all_mean_rewards
        )
        report.add_image(img, 'Mean rewards', width=500)
        
        img = plot_line_graph(
            osp.join(log_dir, 'coverages.png'),
            range(hyperparams.outer_iters), all_coverage
        )
        report.add_image(img, 'Coverages', width=500)
        report.save()

