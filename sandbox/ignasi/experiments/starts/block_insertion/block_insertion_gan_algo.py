from sandbox.ignasi.utils import set_env_no_gpu

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
from sandbox.ignasi.logging import HTMLReport
from sandbox.ignasi.logging import format_dict
from sandbox.ignasi.logging.logger import ExperimentLogger
from sandbox.ignasi.logging.visualization import plot_labeled_states

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from sandbox.ignasi.state.evaluator import label_states
from sandbox.ignasi.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator
from sandbox.ignasi.state.generator import StateGAN
from sandbox.ignasi.envs.start_env import generate_starts
from sandbox.ignasi.state.utils import StateCollection

from sandbox.ignasi.envs.goal_env import GoalExplorationEnv, generate_initial_goals
from sandbox.ignasi.envs.goal_start_env import GoalStartExplorationEnv
from sandbox.ignasi.envs.block_insertion.block_insertion_env import BLOCK_INSERTION_ENVS
from sandbox.ignasi.envs.block_insertion.utils import plot_policy_performance

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])

    inner_env = BLOCK_INSERTION_ENVS[v['env_idx'] - 1]() # TODO: why is this not normalized in goal version?

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

    # TODO: fix all these
    goal_center = np.array((0, 1, 0))
    if 'ultimate_goal' not in v:
        v['ultimate_goal'] = goal_center
    if 'goal_center' not in v:
        v['goal_center'] = goal_center
        # import pdb; pdb.set_trace()
    if 'start_size' not in v:
        v['start_size'] = goal_dim
    if 'start_range' not in v:
        v['start_range'] = 2
    if 'goal_range' not in v:
        v['goal_range'] = 2
    if 'start_center' not in v:
        v['start_center'] = goal_center
    if 'goal_size' not in v:
        v['goal_size'] = v['start_size']
    # TODO: what is v['goal range']? turn on plotting code
    fixed_goal_generator = FixedStateGenerator(state=v['ultimate_goal'])
    uniform_start_generator = UniformStateGenerator(state_size=v['start_size'], bounds=v['start_range'],
                                                    center=v['start_center'])

    env = GoalStartExplorationEnv(
        env=inner_env,
        start_generator=uniform_start_generator,
        obs2start_transform=lambda x: x[:goal_dim],
        goal_generator=fixed_goal_generator,
        obs2goal_transform=lambda x: x[:goal_dim],
        terminal_eps=v['terminal_eps'],
        distance_metric=v['distance_metric'],
        extend_dist_rew=v['extend_dist_rew'],
        only_feasible=v['only_feasible'],
        terminate_env=True,
    )

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(128, 128),
        # Fix the variance since different goals will require different variances, making this parameter hard to learn.
        learn_std=v['learn_std'],
        adaptive_std=v['adaptive_std'],
        std_hidden_sizes=(64, 64),  # this is only used if adaptive_std is true!
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
        state_size=v['start_size'],
        evaluater_size=v['num_labels'],
        state_range=v['start_range'],
        state_center=v['start_center'],
        state_noise_level=v['start_noise_level'],
        generator_layers=v['gan_generator_layers'],
        discriminator_layers=v['gan_discriminator_layers'],
        noise_size=v['gan_noise_size'],
        tf_session=tf_session,
        configs=gan_configs,
    )
    logger.log("pretraining the GAN...")
    if v['smart_init']:
        feasible_starts = generate_starts(env, starts=[v['ultimate_goal']], horizon=50)  # without giving the policy it does brownian mo.
        labels = np.ones((feasible_starts.shape[0], 2)).astype(np.float32)  # make them all good goals
        plot_labeled_states(feasible_starts, labels, report=report, itr=outer_iter,
                            limit=v['goal_range'], center=v['goal_center'])

        dis_loss, gen_loss = gan.pretrain(states=feasible_starts, outer_iters=v['gan_outer_iters'])
        print("Loss of Gen and Dis: ", gen_loss, dis_loss)
    else:
        gan.pretrain_uniform(outer_iters=500, report=report)  # v['gan_outer_iters'])

    # log first samples form the GAN
    initial_starts, _ = gan.sample_states_with_noise(v['num_new_starts'])

    logger.log("Labeling the starts")
    labels = label_states(initial_starts, env, policy, v['horizon'], as_goals=False, n_traj=v['n_traj'], key='goal_reached')

    plot_labeled_states(initial_starts, labels, report=report, itr=outer_iter,
                        limit=v['goal_range'], center=v['goal_center'])
    report.new_row()

    all_starts = StateCollection(distance_threshold=v['coll_eps'])

    for outer_iter in range(1, v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)
        # Sample GAN
        logger.log("Sampling starts from the GAN")
        raw_starts, _ = gan.sample_states_with_noise(v['num_new_starts'])

        if v['replay_buffer'] and outer_iter > 0 and all_starts.size > 0:
            old_starts = all_starts.sample(v['num_old_starts'])
            starts = np.vstack([raw_starts, old_starts])
        else:
            starts = raw_starts

        with ExperimentLogger(log_dir, 'last', snapshot_mode='last', hold_outter_log=True):
            logger.log("Updating the environment start generator")
            env.update_start_generator(
                UniformListStateGenerator(
                    starts.tolist(), persistence=v['persistence'], with_replacement=v['with_replacement'],
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
                discount=v['discount'],
                plot=False,
            )

            algo.train()

        logger.log("Labeling the starts")
        labels = label_states(starts, env, policy, v['horizon'], as_goals=False, n_traj=v['n_traj'], key='goal_reached')

        plot_labeled_states(starts, labels, report=report, itr=outer_iter, limit=v['goal_range'],
                            center=v['goal_center'])

        # ###### extra for deterministic:
        # logger.log("Labeling the goals deterministic")
        # with policy.set_std_to_0():
        #     labels_det = label_states(goals, env, policy, v['horizon'], n_traj=v['n_traj'], n_processes=1)
        # plot_labeled_states(goals, labels_det, report=report, itr=outer_iter, limit=v['goal_range'], center=v['goal_center'])

        labels = np.logical_and(labels[:, 0], labels[:, 1]).astype(int).reshape((-1, 1))

        logger.log("Training the GAN")
        if np.any(labels):
            gan.train(
                starts, labels,
                v['gan_outer_iters'],
            )

        logger.log("Evaluate Unif")
        reward_img, mean_success = plot_policy_performance(policy, env, horizon=v['horizon'], n_traj=3,
                                                           key='goal_reached')

        with logger.tabular_prefix('Outer_'):
            logger.record_tabular('iter', outer_iter)
            # logger.record_tabular('MeanRewards', mean_rewards)
            logger.record_tabular('Success', mean_success)

        report.add_image(
            reward_img,
            'policy performance\n itr: {} \nsuccess: {}'.format(outer_iter, mean_success)
        )

        if v['env_idx'] == 5:
            img, avg_success = plot_policy_performance_sliced(
                policy, env, v['horizon'], slices=(0, None, None), n_traj=3
            )
            report.add_image(img, 'policy_rewards_sliced_{}\nAvg_success: {}'.format(outer_iter, avg_success))

        logger.dump_tabular(with_prefix=False)
        report.new_row()

        # append new goals to list of all goals (replay buffer): Not the low reward ones!!
        filtered_raw_start = [start for start, label in zip(starts, labels) if label[0] == 1]
        all_starts.append(filtered_raw_start)

        if v['add_on_policy']:
            logger.log("sampling on policy")
            feasible_goals = generate_initial_goals(env, policy, goal_bounds, goal_center, horizon=v['horizon'])
            # downsampled_feasible_goals = feasible_goals[np.random.choice(feasible_goals.shape[0], v['add_on_policy']),:]
            all_starts.append(feasible_goals)
