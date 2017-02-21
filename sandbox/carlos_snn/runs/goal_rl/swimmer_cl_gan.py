import os
import os.path as osp
import sys
import time
import random
import numpy as np
import scipy
import tensorflow as tf
import tflearn
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.misc import logger

from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sampler.stateful_pool import singleton_pool
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.misc.instrument import VariantGenerator, variant

from sandbox.young_clgan.lib.envs.base import GoalIdxExplorationEnv
from sandbox.young_clgan.lib.envs.base import UniformListGoalGenerator, FixedGoalGenerator, update_env_goal_generator
from sandbox.young_clgan.lib.goal import *
from sandbox.young_clgan.lib.logging.html_report import HTMLReport
from sandbox.young_clgan.lib.logging.visualization import *
from sandbox.young_clgan.lib.logging.logger import ExperimentLogger

from sandbox.young_clgan.lib.utils import initialize_parallel_sampler
# initialize_parallel_sampler()
# initialize_parallel_sampler(n_processes=-1)

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]

if __name__ == '__main__':

    exp_prefix = 'goalGAN-swimmer'
    vg = VariantGenerator()
    # algorithm params
    vg.add('seed', range(30, 40, 10))
    vg.add('n_itr', [2000])
    vg.add('batch_size', [5000])
    vg.add('max_path_length', [500])
    # # GAN params
    # vg.add('goal', lambda goal_generator: [(0, -1, 0), ] if goal_generator == FixedGoalGenerator else [None])
    # vg.add('goal_reward', ['InverseDistance', 'NegativeDistance'])
    # vg.add('goal_weight', [1, 0])
    # vg.add('terminal_bonus', [1e3, 0])
    # old hyperparams
    vg.add('horizon', [500])
    vg.add('goal_range', [2])
    vg.add('goal_noise_level', [1])
    vg.add('min_reward', [5])
    vg.add('max_reward', [6000])
    vg.add('improvement_threshold', [10])
    vg.add('outer_iters', [50])
    vg.add('inner_iters', [50])
    vg.add('pg_batch_size', [20000])
    vg.add('gan_outer_iters', [5])
    vg.add('gan_discriminator_iters', [200])
    vg.add('gan_generator_iters', [5])
    # gan_configs
    vg.add('GAN_batch_size', [128])  # proble with repeated name!!
    vg.add('GAN_generator_activation', ['relu'])
    vg.add('GAN_discriminator_activation', ['relu'])
    vg.add('GAN_generator_optimizer', [tf.train.AdamOptimizer])
    vg.add('GAN_generator_optimizer_stepSize', [0.001])
    vg.add('GAN_discriminator_optimizer', [tf.train.AdamOptimizer])
    vg.add('GAN_discriminator_optimizer_stepSize', [0.001])
    vg.add('GAN_generator_weight_initializer', [tflearn.initializations.truncated_normal])
    vg.add('GAN_generator_weight_initializer_stddev', [0.05])
    vg.add('GAN_discriminator_weight_initializer', [tflearn.initializations.truncated_normal])
    vg.add('GAN_discriminator_weight_initializer_stddev', [0.02])
    vg.add('GAN_discriminator_batch_noise_stddev', [1e-2])


    def run_task(v):
        random.seed(v['seed'])
        np.random.seed(v['seed'])

        tf_session = tf.Session()

        gan_configs = {key[4:]: value for key, value in v.items() if 'GAN_' in key}
        for key, value in gan_configs.items():
            if value is tf.train.AdamOptimizer:
                gan_configs[key] = tf.train.AdamOptimizer(gan_configs[key+'_stepSize'])
            if value is tflearn.initializations.truncated_normal:
                gan_configs[key] = tflearn.initializations.truncated_normal(stddev=gan_configs[key+'_stddev'])
        print(gan_configs)

        gan = GoalGAN(
            goal_size=2,
            evaluater_size=3,
            goal_range=v['goal_range'],
            goal_noise_level=v['goal_noise_level'],
            generator_layers=[256, 256],
            discriminator_layers=[128, 128],
            noise_size=4,
            tf_session=tf_session,
            configs=gan_configs,
        )

        inner_env = normalize(SwimmerEnv())
        goal_generator = FixedGoalGenerator([0.1, 0.1])
        env = GoalIdxExplorationEnv(env=inner_env, goal_generator=goal_generator, goal_weight=1)  # this goal_generator will be updated by a uniform after

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            # Fix the variance since different goals will require different variances, making this parameter hard to learn.
            learn_std=False
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)


        # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
        log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
        print("******* logdir: ", log_dir)

        img = plot_policy_reward(
            policy, env, v['goal_range'],
            horizon=v['horizon'],
            fname='{}/policy_reward_init.png'.format(log_dir),
        )
        report = HTMLReport(osp.join(log_dir, 'report.html'))

        report.add_header("{}".format(EXPERIMENT_TYPE))
        # report.add_text(format_dict(vg.variant))
        report.add_image(img, 'policy performance initialization\n')

        # Pretrain GAN with uniform distribution on the GAN output space and log a sample
        gan.pretrain_uniform()
        img = plot_gan_samples(gan, v['goal_range'], '{}/start.png'.format(log_dir))
        report.add_image(img, 'GAN pretrained uniform')

        report.save()
        report.new_row()

        all_goals = np.zeros((0, 2))

        print("about to start outer loop")
        for outer_iter in range(v['outer_iters']):

            # Train GAN
            raw_goals, _ = gan.sample_goals_with_noise(2000)

            if outer_iter > 0:
                # sampler uniformly 2000 old goals and add them to the training pool (50/50)
                old_goal_indices = np.random.randint(0, all_goals.shape[0], 2000)
                old_goals = all_goals[old_goal_indices, :]
                goals = np.vstack([raw_goals, old_goals])
            else:
                goals = raw_goals

            # append new goals to list of all goals (replay buffer)
            all_goals = np.vstack([all_goals, raw_goals])

            rewards_before = evaluate_goals(goals, env, policy, v['horizon'])

            with ExperimentLogger(log_dir, outer_iter):
                # set goal generator to uniformly sample from selected all_goals
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
                    batch_size=v['pg_batch_size'],
                    max_path_length=v['horizon'],
                    n_itr=v['inner_iters'],
                    discount=0.995,
                    step_size=0.01,
                    plot=False,
                )

                algo.train()

                img = plot_policy_reward(
                    policy, env,  v['goal_range'],
                    horizon=v['horizon'],
                    # fname='{}/policy_reward_{}.png'.format(log_config.plot_dir, outer_iter),
                )
                report.add_image(img, 'policy performance\n itr: {}'.format(outer_iter))
                report.save()

                # this re-evaluate the final policy in the collection of goals
                labels = label_goals(
                    goals, env, policy, v['horizon'],
                    min_reward=v['min_reward'],
                    max_reward=v['max_reward'],
                    old_rewards=rewards_before,
                    improvement_threshold=v['improvement_threshold']
                )

                gan.train(
                    goals, labels,
                     v['gan_outer_iters'],
                     v['gan_generator_iters'],
                     v['gan_discriminator_iters']
                )

                img = plot_labeled_samples(
                    samples=goals, labels=labels, limit=v['goal_range'] + 5,
                    # '{}/sampled_goals_{}.png'.format(log_dir, outer_iter),  # if i don't give the file it doesn't save
                )
                report.add_image(img, 'goals\n itr: {}'.format(outer_iter), width=500)
                report.save()
                report.new_row()

    for vv in vg.variants():
        run_experiment_lite(
            run_task,
            variant=vv,
            mode='local',
            # n_parallel=n_parallel,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            seed=vv['seed'],
            exp_prefix=exp_prefix,
        )
