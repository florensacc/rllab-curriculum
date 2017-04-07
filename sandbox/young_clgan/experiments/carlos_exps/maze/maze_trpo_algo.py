import os
import os.path as osp
from collections import OrderedDict

from sandbox.young_clgan.envs.maze.maze_evaluate import test_and_plot_policy
from sandbox.young_clgan.envs.maze.point_maze_env import PointMazeEnv
from sandbox.young_clgan.logging import HTMLReport
from sandbox.young_clgan.logging import format_dict
from sandbox.young_clgan.logging.visualization import save_image, plot_labeled_samples, \
    plot_line_graph

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Symbols that need to be stubbed
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

import random

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')

from sandbox.young_clgan.envs.base import UniformListGoalGenerator, FixedGoalGenerator, update_env_goal_generator

from sandbox.young_clgan.logging.logger import ExperimentLogger
from sandbox.young_clgan.goal.evaluator import convert_label, label_goals
from rllab.misc import logger

from sandbox.young_clgan.utils import initialize_parallel_sampler

initialize_parallel_sampler()

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])

    tf_session = tf.Session()

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=2)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    # # GAN
    # logger.log("Instantiating the GAN...")
    # gan_configs = {key[4:]: value for key, value in v.items() if 'GAN_' in key}
    # for key, value in gan_configs.items():
    #     if value is tf.train.AdamOptimizer:
    #         gan_configs[key] = tf.train.AdamOptimizer(gan_configs[key + '_stepSize'])
    #     if value is tflearn.initializations.truncated_normal:
    #         gan_configs[key] = tflearn.initializations.truncated_normal(stddev=gan_configs[key + '_stddev'])
    #
    # gan = GoalGAN(
    #     goal_size=v['goal_size'],
    #     evaluater_size=3,
    #     goal_range=v['goal_range'],
    #     goal_noise_level=v['goal_noise_level'],
    #     generator_layers=v['gan_generator_layers'],
    #     discriminator_layers=v['gan_discriminator_layers'],
    #     noise_size=v['gan_noise_size'],
    #     tf_session=tf_session,
    #     configs=gan_configs,
    # )

    env = normalize(PointMazeEnv(
        goal_generator=FixedGoalGenerator(v['final_goal']),
        reward_dist_threshold=v['reward_dist_threshold'],
        indicator_reward=v['indicator_reward'],
        terminal_eps=v['terminal_eps'],
    ))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        # Fix the variance since different goals will require different variances, making this parameter hard to learn.
        learn_std=v['learn_std'],
        output_gain=v['output_gain'],
        init_std=v['policy_init_std'],
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    # n_traj = 3 if v['indicator_reward'] else 1
    n_traj = 3
    sampling_res = 2
    # test_and_plot_policy(policy, env, max_reward=v['max_reward'], n_traj=n_traj)
    # reward_img = save_image(fname=osp.join(log_dir, 'policy_reward_init.png'))
    # report.add_image(
    #     reward_img,
    #     'policy performance initialization\n'
    # )

    # logger.log("Pretraining the gan for uniform sampling")
    # gan.pretrain_uniform()

    # logger.log("pretraining the GAN...")
    # if v['smart_init']:
    #     gan.pretrain(
    #         generate_initial_goals(env, policy, v['goal_range'], horizon=v['horizon']),
    #         outer_iters=30, generator_iters=10, discriminator_iters=200,
    #     )
    # else:
    #     gan.pretrain_uniform()

    # logger.log("Plotting GAN samples")
    # img = plot_gan_samples(gan, v['goal_range'], osp.join(log_dir, 'start.png'))
    # report.add_image(img, 'GAN initialization')

    report.save()
    report.new_row()

    all_mean_rewards = []
    all_coverage = []
    all_success = []

    for outer_iter in range(v['outer_iters']):

        # # Train GAN
        # logger.log("Sampling goals from the GAN")
        # raw_goals, _ = gan.sample_goals_with_noise(v['num_new_goals'])
        #
        # if outer_iter > 0 and all_goals.size > 0:
        #     old_goals = all_goals.sample(v['num_old_goals'])
        #     goals = np.vstack([raw_goals, old_goals])
        # else:
        #     goals = raw_goals

        goals = np.random.uniform(-v['goal_range'], v['goal_range'], size=(300, v['goal_size']))

        with ExperimentLogger(log_dir, outer_iter, snapshot_mode='last', hold_outter_log=True):
            logger.log("Updating the environment goal generator")
            if v['unif_goals']:
                update_env_goal_generator(
                    env,
                    UniformListGoalGenerator(goals.tolist())
                )
            else:
                update_env_goal_generator(env, FixedGoalGenerator(v['final_goal']))

            logger.log("Training the algorithm")
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=v['pg_batch_size'],
                max_path_length=v['horizon'],
                n_itr=v['inner_iters'],
                discount=v['discount'],
                step_size=0.01,
                plot=False,
                gae_lambda=v['gae_lambda'])

            algo.train()

        logger.log('Generating the Heatmap...')
        avg_rewards, avg_success, heatmap = test_and_plot_policy(policy, env, max_reward=v['max_reward'],
                                                                 sampling_res=sampling_res, n_traj=n_traj)
        reward_img = save_image()

        mean_rewards = np.mean(avg_rewards)
        # coverage is more restrictive! In avg it has to be above max_reward: if some are but not the avg, not count!
        coverage = np.mean([int(avg_reward >= v['max_reward']) for avg_reward in avg_rewards])
        success = np.mean(avg_success)

        all_mean_rewards.append(mean_rewards)
        all_coverage.append(coverage)
        all_success.append(success)

        with logger.tabular_prefix('Outer_'):
            logger.record_tabular('MeanRewards', mean_rewards)
            logger.record_tabular('Coverage', coverage)
            logger.record_tabular('Success', success)
        # logger.dump_tabular(with_prefix=False)

        report.add_image(
            reward_img,
            'policy performance\n itr: {} \nmean_rewards: {} \ncoverage: {}\nsuccess: {}'.format(
                outer_iter, all_mean_rewards[-1],
                all_coverage[-1], all_success[-1],
            )
        )

        # plt.scatter(goals[:, 0], goals[:, 1])
        # scatter_plot = save_image()
        # report.add_image(scatter_plot, 'goals sampled for itr{}'.format(outer_iter))

        report.save()

        logger.log("Labeling the goals")
        labels = label_goals(
            goals, env, policy, v['horizon'],
            min_reward=v['min_reward'],
            max_reward=v['max_reward'],
            old_rewards=None,
            improvement_threshold=v['improvement_threshold'],
            n_traj=n_traj)

        logger.log("Converting the labels")
        goal_classes, text_labels = convert_label(labels)

        logger.log("Plotting the labeled samples")
        total_goals = labels.shape[0]
        goal_class_frac = OrderedDict()  # this needs to be an ordered dict!! (for the log tabular)
        for k in text_labels.keys():
            frac = np.sum(goal_classes == k) / total_goals
            logger.record_tabular('GenGoal_frac_' + text_labels[k], frac)
            goal_class_frac[text_labels[k]] = frac

        img = plot_labeled_samples(
            samples=goals, sample_classes=goal_classes, text_labels=text_labels, limit=v['goal_range'],
            # '{}/sampled_goals_{}.png'.format(log_dir, outer_iter),  # if i don't give the file it doesn't save
        )
        summary_string = ''
        for key, value in goal_class_frac.items():
            summary_string += key + ' frac: ' + str(value) + '\n'
        report.add_image(img, 'itr: {}\nLabels of generated goals:\n{}'.format(outer_iter, summary_string), width=500)

        logger.dump_tabular(with_prefix=False)
        report.save()
        report.new_row()

        # # append new goals to list of all goals (replay buffer): Not the low reward ones!!
        # filtered_raw_goals = [goal for goal, label in zip(goals, labels) if label[0] == 1]
        # all_goals.append(filtered_raw_goals)

    img = plot_line_graph(
        osp.join(log_dir, 'mean_rewards.png'),
        range(v['outer_iters']), all_mean_rewards
    )
    report.add_image(img, 'Mean rewards', width=500)

    img = plot_line_graph(
        osp.join(log_dir, 'coverages.png'),
        range(v['outer_iters']),
    )

    report.add_image(img, 'Coverages', width=500)
    report.save()
