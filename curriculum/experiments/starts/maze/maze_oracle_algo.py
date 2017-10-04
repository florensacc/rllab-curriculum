import matplotlib

matplotlib.use('Agg')
import os
import os.path as osp
import multiprocessing
import random
import numpy as np
import tensorflow as tf
import tflearn
from collections import OrderedDict

from rllab.misc import logger
from curriculum.logging import HTMLReport
from curriculum.logging import format_dict
from curriculum.logging.logger import ExperimentLogger
from curriculum.logging.visualization import save_image, plot_labeled_samples, plot_labeled_states

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from curriculum.state.evaluator import convert_label, label_states, evaluate_states
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator
from curriculum.state.utils import StateCollection

from curriculum.envs.goal_start_env import GoalStartExplorationEnv
from curriculum.envs.maze.maze_evaluate import test_and_plot_policy, sample_unif_feas, unwrap_maze, plot_policy_means
from curriculum.envs.maze.point_maze_env import PointMazeEnv

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    sampling_res = 2 if 'sampling_res' not in v.keys() else v['sampling_res']
    samples_per_cell = 10  # for the oracle rejection sampling

    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=3)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    inner_env = normalize(PointMazeEnv(maze_id=v['maze_id']))

    fixed_goal_generator = FixedStateGenerator(state=v['ultimate_goal'])
    uniform_start_generator = UniformStateGenerator(state_size=v['start_size'], bounds=v['start_range'],
                                                    center=v['start_center'])

    env = GoalStartExplorationEnv(
        env=inner_env,
        start_generator=uniform_start_generator,
        goal_generator=fixed_goal_generator,
        obs2start_transform=lambda x: x[:v['start_size']],
        obs2goal_transform=lambda x: x[:v['goal_size']],
        terminal_eps=v['terminal_eps'],
        distance_metric=v['distance_metric'],
        extend_dist_rew=v['extend_dist_rew'],
        only_feasible=v['only_feasible'],
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

    # initialize all logging arrays on itr0
    outer_iter = 0

    logger.log('Generating the Initial Heatmap...')
    plot_policy_means(policy, env, sampling_res=2, report=report, limit=v['start_range'], center=v['start_center'])
    test_and_plot_policy(policy, env, as_goals=False, max_reward=v['max_reward'], sampling_res=sampling_res, n_traj=v['n_traj'],
                         itr=outer_iter, report=report, center=v['start_center'], limit=v['start_range'])
    report.new_row()

    all_starts = StateCollection(distance_threshold=v['coll_eps'])
    total_rollouts = 0

    for outer_iter in range(1, v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)
        logger.log("Sampling starts")

        starts = np.array([]).reshape((-1, v['start_size']))
        k = 0
        while starts.shape[0] < v['num_new_starts']:
            print('good starts collected: ', starts.shape[0])
            logger.log("Sampling and labeling the starts: %d" % k)
            k += 1
            unif_starts = sample_unif_feas(env, samples_per_cell=samples_per_cell)
            if v['start_size'] > 2:
                unif_starts = np.array([np.concatenate([start, np.random.uniform(-v['start_range'], v['start_range'], 2)])
                               for start in unif_starts])
            labels = label_states(unif_starts, env, policy, v['horizon'],
                                  as_goals=False, n_traj=v['n_traj'], key='goal_reached')
            # plot_labeled_states(unif_starts, labels, report=report, itr=outer_iter, limit=v['start_range'],
            #                     center=v['start_center'], maze_id=v['maze_id'])
            logger.log("Converting the labels")
            init_classes, text_labels = convert_label(labels)
            starts = np.concatenate([starts, unif_starts[init_classes == 2]]).reshape((-1, v['start_size']))

        if v['replay_buffer'] and outer_iter > 0 and all_starts.size > 0:
            old_starts = all_starts.sample(v['num_old_starts'])
            starts = np.vstack([starts, old_starts])
        # report.new_row()

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
                gae_lambda=v['gae_lambda'],
                plot=False,
            )

            algo.train()

        logger.log('Generating the Heatmap...')
        plot_policy_means(policy, env, sampling_res=2, report=report, limit=v['start_range'], center=v['start_center'])
        test_and_plot_policy(policy, env, as_goals=False, max_reward=v['max_reward'], sampling_res=sampling_res, n_traj=v['n_traj'],
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

        # rollouts used for labeling (before TRPO itrs):
        num_empty_spaces = len(unwrap_maze(env).find_empty_space())
        logger.record_tabular('LabelingRollouts', k * v['n_traj'] * samples_per_cell * num_empty_spaces)
        total_rollouts += k * v['n_traj'] * samples_per_cell * num_empty_spaces
        logger.record_tabular('TotalLabelingRollouts', total_rollouts)

        logger.dump_tabular(with_prefix=False)
        report.new_row()

        # append new goals to list of all goals (replay buffer): Not the low reward ones!!
        filtered_raw_starts = [start for start, label in zip(starts, labels) if label[0] == 1]
        all_starts.append(filtered_raw_starts)
