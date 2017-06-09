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
from sandbox.young_clgan.envs.base import UniformListStateGenerator, update_env_state_generator, UniformStateGenerator, \
    FixedStateGenerator
from sandbox.young_clgan.state.generator import StateGAN
from sandbox.young_clgan.state.utils import StateCollection

from sandbox.young_clgan.envs.goal_env import GoalExplorationEnv, generate_initial_goals
from sandbox.young_clgan.envs.start_env import update_env_start_generator
from sandbox.young_clgan.envs.goal_start_env import GoalStartExplorationEnv
from sandbox.young_clgan.envs.maze.maze_evaluate import test_and_plot_policy, plot_policy_means
from sandbox.young_clgan.envs.maze.point_maze_env import PointMazeEnv

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    sampling_res = 2 if 'sampling_res' not in v.keys() else v['sampling_res']

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
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
        obs_transform=lambda x: x[:int(len(x) / 2)],
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
                         itr=outer_iter, report=report, limit=v['start_range'], center=v['start_center'])
    report.new_row()

    for outer_iter in range(1, v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)
        logger.log("Sampling goals from the GAN")
        starts = np.random.uniform(np.array(v['start_center']) - np.array(v['start_range']),
                                   np.array(v['start_center']) + np.array(v['start_range']), size=(300, v['start_size']))

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

        logger.log('Generating the Heatmap...')
        plot_policy_means(policy, env, sampling_res=2, report=report, limit=v['start_range'], center=v['start_center'])
        test_and_plot_policy(policy, env, as_goals=False, max_reward=v['max_reward'], sampling_res=sampling_res, n_traj=v['n_traj'],
                             itr=outer_iter, report=report, limit=v['goal_range'], center=v['goal_center'])

        logger.log("Labeling the goals")
        labels = label_states(starts, env, policy, v['horizon'], as_goals=False, n_traj=v['n_traj'], key='goal_reached')

        plot_labeled_states(starts, labels, report=report, itr=outer_iter, limit=v['goal_range'],
                            center=v['goal_center'], maze_id=v['maze_id'])

        # ###### extra for deterministic:
        # logger.log("Labeling the goals deterministic")
        # with policy.set_std_to_0():
        #     labels_det = label_states(goals, env, policy, v['horizon'], n_traj=v['n_traj'], n_processes=1)
        # plot_labeled_states(goals, labels_det, report=report, itr=outer_iter, limit=v['goal_range'], center=v['goal_center'])

        logger.dump_tabular(with_prefix=False)
        report.new_row()
