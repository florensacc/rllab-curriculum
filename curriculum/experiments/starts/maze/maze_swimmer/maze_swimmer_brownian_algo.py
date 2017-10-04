import matplotlib
import cloudpickle
import pickle

from curriculum.logging.visualization import plot_labeled_states

matplotlib.use('Agg')
import os
import os.path as osp
import random
import time
import numpy as np

from rllab.misc import logger
from collections import OrderedDict
from curriculum.logging import HTMLReport
from curriculum.logging import format_dict
from curriculum.logging.logger import ExperimentLogger

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from rllab import config
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from curriculum.state.evaluator import convert_label, label_states, evaluate_states, label_states_from_paths
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator
from curriculum.state.utils import StateCollection

from curriculum.envs.start_env import generate_starts, find_all_feasible_states, find_all_feasible_states_plotting
from curriculum.envs.goal_start_env import GoalStartExplorationEnv
from curriculum.envs.arm3d.arm3d_disc_env import Arm3dDiscEnv
from curriculum.envs.maze.maze_swim.swim_maze_env import SwimmerMazeEnv
from curriculum.envs.maze.maze_evaluate import test_and_plot_policy, sample_unif_feas, unwrap_maze, \
    plot_policy_means


EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    logger.log("Initializing report...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    if log_dir is None:
        log_dir = "/home/michael/"
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=4)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    inner_env = normalize(SwimmerMazeEnv())

    fixed_goal_generator = FixedStateGenerator(state=v['ultimate_goal'])
    fixed_start_generator = FixedStateGenerator(state=v['ultimate_goal'])

    env = GoalStartExplorationEnv(
        env=inner_env,
        start_generator=fixed_start_generator,
        obs2start_transform=lambda x: x[:v['start_size']],
        goal_generator=fixed_goal_generator,
        obs2goal_transform=lambda x: x[-3:-1],
        terminal_eps=v['terminal_eps'],
        distance_metric=v['distance_metric'],
        extend_dist_rew=v['extend_dist_rew'],
        inner_weight=v['inner_weight'],
        goal_weight=v['goal_weight'],
        terminate_env=True,
    )

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        # Fix the variance since different goals will require different variances, making this parameter hard to learn.
        learn_std=v['learn_std'],
        adaptive_std=v['adaptive_std'],
        std_hidden_sizes=(16, 16),  # this is only used if adaptive_std is true!
        output_gain=v['output_gain'],
        init_std=v['policy_init_std'],
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    # load the state collection from data_upload
    load_dir = 'data_upload/state_collections/'
    # all_feasible_starts = pickle.load(open(osp.join(config.PROJECT_PATH, load_dir, 'all_feasible_states_min.pkl'), 'rb'))
    # print("we have %d feasible starts" % all_feasible_starts.size)

    all_starts = StateCollection(distance_threshold=v['coll_eps'])
    # brownian_starts = StateCollection(distance_threshold=v['regularize_starts'])
    # with env.set_kill_outside():
    seed_starts = generate_starts(env, starts=[v['start_goal']], horizon=v['initial_brownian_horizon'], size=5000, # size speeds up training a bit
                                  variance=v['brownian_variance'], subsample=v['num_new_starts'])  # , animated=True, speedup=1)
    np.random.shuffle(seed_starts)

    # with env.set_kill_outside():
    feasible_states = find_all_feasible_states_plotting(env, seed_starts, distance_threshold=1, brownian_variance=1, animate=True)

    # print("hi")
    # show where these states are:
    # shuffled_starts = np.array(seed_starts.state_list)
    # np.random.shuffle(shuffled_starts)
    # generate_starts(env, starts=seed_starts, horizon=100, variance=v['brownian_variance'], animated=True, speedup=10)

    if 'gae_lambda' not in v:
        v['gae_lambda'] = 1
    for outer_iter in range(1, v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)
        logger.log("Sampling starts")

        starts = generate_starts(env, starts=seed_starts, subsample=v['num_new_starts'], size=5000,
                                 horizon=v['brownian_horizon'], variance=v['brownian_variance'])
        labels = label_states(starts, env, policy, v['horizon'],
                              as_goals=False, n_traj=v['n_traj'], key='goal_reached')
        plot_labeled_states(starts, labels, report=report, itr=outer_iter, limit=v['goal_range'],
                            center=v['goal_center'], maze_id=v['maze_id'],
                            summary_string_base='initial starts labels:\n')
        report.save()

        if v['replay_buffer'] and outer_iter > 0 and all_starts.size > 0:
            old_starts = all_starts.sample(v['num_old_starts'])
            starts = np.vstack([starts, old_starts])

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
                gae_lambda=v['gae_lambda'],
                step_size=0.01,
                discount=v['discount'],
                plot=False,
            )

            algo.train()

        logger.log('Generating the Heatmap...')

        # policy means should not mean too much
        # plot_policy_means(policy, env, sampling_res=2, report=report, limit=v['goal_range'], center=v['goal_center'])
        test_and_plot_policy(policy, env, as_goals=False, max_reward=v['max_reward'], sampling_res=1,
                             n_traj=v['n_traj'],
                             itr=outer_iter, report=report, center=v['goal_center'], limit=v['goal_range'])

        logger.log("Labeling the starts")
        labels = label_states(starts, env, policy, v['horizon'], as_goals=False, n_traj=v['n_traj'], key='goal_reached')

        plot_labeled_states(starts, labels, report=report, itr=outer_iter, limit=v['goal_range'],
                            center=v['goal_center'], maze_id=v['maze_id'])

        # ###### extra for deterministic:
        # logger.log("Labeling the goals deterministic")
        # with policy.set_std_to_0():
        #     labels_det = label_states(goals, env, policy, v['horizon'], n_traj=v['n_traj'], n_processes=1)
        # plot_labeled_states(goals, labels_det, report=report, itr=outer_iter, limit=v['goal_range'], center=v['goal_center'])

        labels = np.logical_and(labels[:, 0], labels[:, 1]).astype(int).reshape((-1, 1))

        logger.dump_tabular(with_prefix=False)
        report.new_row()

        # append new states to list of all starts (replay buffer): Not the low reward ones!!
        filtered_raw_starts = [start for start, label in zip(starts, labels) if label[0] == 1]
        if len(filtered_raw_starts) > 0:  # add a tone of noise if all the states I had ended up being high_reward!
            seed_starts = filtered_raw_starts
        else:
            seed_starts = generate_starts(env, starts=starts, horizon=v['horizon'] * 2, subsample=v['num_new_starts'], size=5000,
                                          variance=v['brownian_variance'] * 10)
        all_starts.append(filtered_raw_starts)

