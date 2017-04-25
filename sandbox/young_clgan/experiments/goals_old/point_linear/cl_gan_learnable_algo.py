import os
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''

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
from sandbox.young_clgan.envs.point_env import PointEnv
from sandbox.young_clgan.goal import *
from sandbox.young_clgan.logging import *

EXPERIMENT_TYPE = 'cl_gan_learnable'



class CLGANPointEnvLinear(RLAlgorithm):

    def __init__(self, hyperparams):
        self.hyperparams = AttrDict(hyperparams)
        
    def convert_label(self, labels):
        # Put good goals last so they will be plotted on top of other goals and be most visible.
        classes = {
            0: 'Bad',
            1: 'Learnable',
        }
        new_labels = np.zeros(labels.shape[0], dtype=int)
        new_labels[labels[:, 2] == True] = 1
    
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

        log_config = format_experiment_log_path(
            __file__, EXPERIMENT_TYPE
        )
        make_log_dirs(log_config)

        random.seed(int(time.time()))
        np.random.seed(int(time.time()))

        report = HTMLReport(log_config.report_file)

        report.add_header("{}, {}".format(EXPERIMENT_TYPE, log_config.experiment_date_host))
        report.add_text(format_dict(hyperparams))

        tf_session = tf.Session()

        gan = GoalGAN(
            goal_size=hyperparams.goal_size,
            evaluater_size=1,
            goal_range=hyperparams.goal_range,
            goal_noise_level=hyperparams.goal_noise_level,
            generator_layers=hyperparams.gan_generator_layers,
            discriminator_layers=hyperparams.gan_discriminator_layers,
            noise_size=hyperparams.gan_noise_size,
            tf_session=tf_session,
            configs=gan_configs
        )

        env = normalize(PointEnv(
            FixedGoalGenerator([0.1, 0.1])
        ))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            # Fix the variance since different goals will require different variances, making this parameter hard to learn.
            learn_std=False
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        img = plot_policy_reward(
            policy, env, hyperparams.goal_range,
            horizon=hyperparams.horizon,
            fname='{}/policy_reward_init.png'.format(log_config.plot_dir),
        )
        report.add_image(img, 'policy performance initialization\n')

        # Pretrain GAN with uniform distribution on the GAN output space
        gan.pretrain_uniform()

        img = plot_gan_samples(gan, hyperparams.goal_range, '{}/start.png'.format(log_config.plot_dir))
        report.add_image(img, 'GAN pretrained uniform')

        report.save()
        report.new_row()

        all_goals = np.zeros((0, 2))
        
        all_mean_rewards = []
        all_coverage = []

        for outer_iter in range(hyperparams.outer_iters):

            # Train GAN
            raw_goals, _ = gan.sample_states_with_noise(2000)

            if outer_iter > 0:
                old_goal_indices = np.random.randint(0, all_goals.shape[0], 2000)
                old_goals = all_goals[old_goal_indices, :]
                goals = np.vstack([raw_goals, old_goals])
            else:
                goals = raw_goals

            all_goals = np.vstack([all_goals, raw_goals])


            rewards_before = evaluate_goals(goals, env, policy, hyperparams.horizon)


            with ExperimentLogger(log_config.log_dir, outer_iter):
                update_env_goal_generator(
                    env,
                    UniformListGoalGenerator(
                        goals.tolist()
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
                    plot=False,
                )

                algo.train()

                img, rewards = plot_policy_reward(
                    policy, env, hyperparams.goal_range,
                    horizon=hyperparams.horizon,
                    fname='{}/policy_reward_{}.png'.format(log_config.plot_dir, outer_iter),
                    return_rewards=True
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

                labels = label_goals(
                    goals, env, policy, hyperparams.horizon,
                    min_reward=hyperparams.min_reward,
                    max_reward=hyperparams.max_reward,
                    old_rewards=rewards_before,
                    improvement_threshold=hyperparams.improvement_threshold
                )

                gan.train(
                    goals, labels[:, 2].reshape(-1, 1),
                    hyperparams.gan_outer_iters,
                    hyperparams.gan_generator_iters,
                    hyperparams.gan_discriminator_iters
                )

                plot_labels, classes = self.convert_label(labels)
                img = plot_labeled_samples(
                    samples=goals, sample_classes=plot_labels,
                    text_labels=classes, limit=hyperparams.goal_range + 5,
                    fname='{}/sampled_goals_{}.png'.format(log_config.plot_dir, outer_iter),
                )
                report.add_image(img, 'goals\n itr: {}'.format(outer_iter), width=500)
                report.save()
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