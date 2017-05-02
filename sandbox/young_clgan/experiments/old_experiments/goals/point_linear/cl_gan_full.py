import os
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''

# from sandbox.young_clgan.lib.utils import initialize_parallel_sampler
# initialize_parallel_sampler()

# Symbols that need to be stubbed
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

from sandbox.young_clgan.lib.envs.base import UniformListStateGenerator, FixedStateGenerator, update_env_state_generator
from sandbox.young_clgan.envs.point_env import PointEnv
from sandbox.young_clgan.goal.evaluator import convert_label, evaluate_goals, label_goals
from sandbox.young_clgan.goal import GoalGAN
from sandbox.young_clgan.logging import *

EXPERIMENT_TYPE = 'cl_gan'


if __name__ == '__main__':

    log_config = format_experiment_log_path(__file__, EXPERIMENT_TYPE)
    make_log_dirs(log_config)

    random.seed(int(time.time()))
    np.random.seed(int(time.time()))

    report = HTMLReport(log_config.report_file)

    hyperparams = AttrDict(
        horizon=200,
        goal_range=15,
        goal_noise_level=1,
        min_reward=5,
        max_reward=6000,
        improvement_threshold=10,
        outer_iters=50,
        inner_iters=5,
        pg_batch_size=20000,
        gan_outer_iters=5,
        gan_discriminator_iters=200,
        gan_generator_iters=5,
    )

    report.add_header("{}, {}".format(EXPERIMENT_TYPE, log_config.experiment_date_host))
    report.add_text(format_dict(hyperparams))

    tf_session = tf.Session()


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

    hyperparams.gan_configs = gan_configs

    gan = GoalGAN(
        goal_size=2,
        evaluater_size=3,
        goal_range=hyperparams.goal_range,
        goal_noise_level=hyperparams.goal_noise_level,
        generator_layers=[256, 256],
        discriminator_layers=[128, 128],
        noise_size=4,
        tf_session=tf_session,
        configs=gan_configs
    )

    env = normalize(PointEnv(
        FixedStateGenerator([0.1, 0.1])
    ))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32),
        # Fix the variance since different goals will require different variances, making this parameter hard to learn.
        learn_std=False
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    # img = plot_policy_reward(
    #     policy, env, hyperparams.goal_range,
    #     horizon=hyperparams.horizon,
    #     fname='{}/policy_reward_init.png'.format(log_config.plot_dir),
    # )
    # report.add_image(img, 'policy performance initialization\n')

    # Pretrain GAN with uniform distribution on the GAN output space
    gan.pretrain_uniform()

    # img = plot_gan_samples(gan, hyperparams.goal_range, '{}/start.png'.format(log_config.plot_dir))
    # report.add_image(img, 'GAN pretrained uniform')

    report.save()
    report.new_row()

    all_goals = np.zeros((0, 2))

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
            update_env_state_generator(
                env,
                UniformListStateGenerator(
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

            img = plot_policy_reward(
                policy, env, hyperparams.goal_range,
                horizon=hyperparams.horizon,
                fname='{}/policy_reward_{}.png'.format(log_config.plot_dir, outer_iter),
            )
            report.add_image(img, 'policy performance\n itr: {}'.format(outer_iter))
            report.save()

            labels = label_goals(
                goals, env, policy, hyperparams.horizon,
                min_reward=hyperparams.min_reward,
                max_reward=hyperparams.max_reward,
                old_rewards=rewards_before,
                improvement_threshold=hyperparams.improvement_threshold
            )

            gan.train(
                goals, labels,
                hyperparams.gan_outer_iters,
                hyperparams.gan_generator_iters,
                hyperparams.gan_discriminator_iters
            )

            plot_labels, classes = convert_label(labels)
            img = plot_labeled_samples(
                goals, plot_labels,
                classes, hyperparams.goal_range + 5,
                '{}/sampled_goals_{}.png'.format(log_config.plot_dir, outer_iter),
            )
            report.add_image(img, 'goals\n itr: {}'.format(outer_iter), width=500)
            report.save()
            report.new_row()
