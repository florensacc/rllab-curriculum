from sandbox.young_clgan.utils import set_env_no_gpu

set_env_no_gpu()

import matplotlib
matplotlib.use('Agg')
import os
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tflearn

from rllab.misc import logger
from sandbox.young_clgan.logging import HTMLReport
from sandbox.young_clgan.logging import format_dict
from sandbox.young_clgan.logging.logger import ExperimentLogger
from sandbox.young_clgan.logging.visualization import plot_labeled_states

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from sandbox.young_clgan.state.evaluator import label_states
from sandbox.young_clgan.envs.base import UniformListStateGenerator, update_env_state_generator, UniformStateGenerator
from sandbox.young_clgan.state.generator import StateGAN
from sandbox.young_clgan.state.utils import StateCollection

from sandbox.young_clgan.envs.goal_env import GoalExplorationEnv, generate_initial_goals
from sandbox.young_clgan.envs.block_insertion.block_insertion_env import BlockInsertionEnv1, BlockInsertionEnv2, \
    BlockInsertionEnv3
from sandbox.young_clgan.envs.block_insertion.utils import plot_policy_performance

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])

    # goal generators
    logger.log("Initializing the goal generators and the inner env...")
    if v['env_idx'] == 1:
        inner_env = normalize(BlockInsertionEnv1())
    elif v['env_idx'] == 2:
        inner_env = normalize(BlockInsertionEnv2())
    else:
        inner_env = normalize(BlockInsertionEnv3())

    goal_center = (inner_env.goal_ub + inner_env.goal_lb) / 2
    goal_bounds = (inner_env.goal_ub - inner_env.goal_lb) / 2
    goal_lb = inner_env.goal_lb
    goal_ub = inner_env.goal_ub
    goal_dim = inner_env.goal_dim

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=3)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    tf_session = tf.Session()

    uniform_goal_generator = UniformStateGenerator(state_size=goal_dim, bounds=[goal_lb, goal_ub])

    env = GoalExplorationEnv(
        env=inner_env, goal_generator=uniform_goal_generator,
        obs_transform=lambda x: x[:goal_dim],
        terminal_eps=v['terminal_eps'],
        distance_metric=v['distance_metric'],
        extend_dist_rew=v['extend_dist_rew'],
        terminate_env=True,
        goal_bounds=goal_bounds,
    )

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32),
        # Fix the variance since different goals will require different variances, making this parameter hard to learn.
        learn_std=v['learn_std'],
        adaptive_std=v['adaptive_std'],
        std_hidden_sizes=(16, 16),  # this is only used if adaptive_std is true!
        output_gain=v['output_gain'],
        init_std=v['policy_init_std'],
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    # initialize all logging arrays on itr0
    outer_iter = 0

    # GAN
    logger.log("Instantiating the GAN...")
    gan_configs = {key[4:]: value for key, value in v.items() if 'GAN_' in key}
    for key, value in gan_configs.items():
        if value is tf.train.AdamOptimizer:
            gan_configs[key] = tf.train.AdamOptimizer(gan_configs[key + '_stepSize'])
        if value is tflearn.initializations.truncated_normal:
            gan_configs[key] = tflearn.initializations.truncated_normal(stddev=gan_configs[key + '_stddev'])

    gan = StateGAN(
        state_size=goal_dim,
        evaluater_size=v['num_labels'],
        state_range=goal_bounds,
        state_center=goal_center,
        state_noise_level=v['goal_noise_level'],
        generator_layers=v['gan_generator_layers'],
        discriminator_layers=v['gan_discriminator_layers'],
        noise_size=v['gan_noise_size'],
        tf_session=tf_session,
        configs=gan_configs,
    )
    logger.log("pretraining the GAN...")
    if v['smart_init']:
        feasible_goals = generate_initial_goals(env, policy, goal_bounds, goal_center=goal_center,
                                                horizon=v['horizon'])
        labels = np.ones((feasible_goals.shape[0], 2)).astype(np.float32)  # make them all good goals

        plot_labeled_states(feasible_goals, labels, report=report, itr=outer_iter,
                            center=goal_center)

        dis_loss, gen_loss = gan.pretrain(states=feasible_goals, outer_iters=v['gan_outer_iters'])
        print("Loss of Gen and Dis: ", gen_loss, dis_loss)
    else:
        gan.pretrain_uniform()

    # log first samples form the GAN
    initial_goals, _ = gan.sample_states_with_noise(v['num_new_goals'])

    logger.log("Labeling the goals")
    labels = label_states(initial_goals, env, policy, v['horizon'], n_traj=v['n_traj'], key='goal_reached')

    plot_labeled_states(initial_goals, labels, report=report, itr=outer_iter,
                        center=goal_center)
    report.new_row()

    all_goals = StateCollection(distance_threshold=v['coll_eps'])

    for outer_iter in range(1, v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)
        # Sample GAN
        logger.log("Sampling goals from the GAN")
        raw_goals, _ = gan.sample_states_with_noise(v['num_new_goals'])

        if v['replay_buffer'] and outer_iter > 0 and all_goals.size > 0:
            old_goals = all_goals.sample(v['num_old_goals'])
            goals = np.vstack([raw_goals, old_goals])
        else:
            goals = raw_goals

        with ExperimentLogger(log_dir, 'last', snapshot_mode='last', hold_outter_log=True):
            logger.log("Updating the environment goal generator")
            update_env_state_generator(
                env,
                UniformListStateGenerator(
                    goals.tolist(), persistence=v['persistence'], with_replacement=v['with_replacement'],
                )
            )

            logger.log("Training the algorithm")
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=v['pg_batch_size'],
                max_path_length=v['horizon'],
                n_itr=v['inner_iters'],
                step_size=0.01,
                plot=False,
            )

            algo.train()

        logger.log("Labeling the goals")
        labels = label_states(goals, env, policy, v['horizon'], n_traj=v['n_traj'], key='goal_reached')

        plot_labeled_states(goals, labels, report=report, itr=outer_iter, center=goal_center)

        labels = np.logical_and(labels[:, 0], labels[:, 1]).astype(int).reshape((-1, 1))

        logger.log("Training the GAN")
        gan.train(
            goals, labels,
            v['gan_outer_iters'],
        )

        logger.log("Evaluate Unif")
        reward_img, mean_success = plot_policy_performance(policy, env, horizon=v['horizon'], n_traj=3, key='goal_reached')

        with logger.tabular_prefix('Outer_'):
            logger.record_tabular('iter', outer_iter)
            # logger.record_tabular('MeanRewards', mean_rewards)
            logger.record_tabular('Success', mean_success)

        report.add_image(
            reward_img,
            'policy performance\n itr: {} \nsuccess: {}'.format(outer_iter,  mean_success)
        )

        logger.dump_tabular(with_prefix=False)
        report.new_row()

        # append new goals to list of all goals (replay buffer): Not the low reward ones!!
        filtered_raw_goals = [goal for goal, label in zip(goals, labels) if label[0] == 1]
        all_goals.append(filtered_raw_goals)

        if v['add_on_policy']:
            logger.log("sampling on policy")
            feasible_goals = generate_initial_goals(env, policy, goal_bounds, goal_center, horizon=v['horizon'])
            # downsampled_feasible_goals = feasible_goals[np.random.choice(feasible_goals.shape[0], v['add_on_policy']),:]
            all_goals.append(feasible_goals)
