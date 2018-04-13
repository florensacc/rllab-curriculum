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
from curriculum.logging.visualization import plot_labeled_states

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline

from curriculum.state.evaluator import label_states, label_states_from_paths
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, \
    FixedStateGenerator
from curriculum.state.generator import StateGAN
from curriculum.state.utils import StateCollection

from curriculum.envs.goal_env import GoalExplorationEnv, generate_initial_goals
from curriculum.envs.maze.maze_evaluate import test_and_plot_policy  # TODO: make this external to maze env
# from curriculum.envs.maze.maze_ant.ant_maze_env import AntMazeEnv
from curriculum.envs.maze.maze_ant.ant_target_env import AntEnv

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    sampling_res = 0 if 'sampling_res' not in v.keys() else v['sampling_res']

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=3)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    inner_env = normalize(AntEnv())

    uniform_goal_generator = UniformStateGenerator(state_size=v['goal_size'], bounds=v['goal_range'],
                                                   center=v['goal_center'])
    env = GoalExplorationEnv(
        env=inner_env, goal_generator=uniform_goal_generator,
        obs2goal_transform=lambda x: x[-3:-1],
        terminal_eps=v['terminal_eps'],
        distance_metric=v['distance_metric'],
        extend_dist_rew=v['extend_dist_rew'],
        append_transformed_obs=v['append_transformed_obs'],
        append_extra_info=v['append_extra_info'],
        # only_feasible=v['only_feasible'],
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
    if v['baseline'] == 'g_mlp':
        baseline = GaussianMLPBaseline(env_spec=env.spec)

    # initialize all logging arrays on itr0
    outer_iter = 0

    logger.log('Generating the Initial Heatmap...')
    test_and_plot_policy(policy, env, max_reward=v['max_reward'], sampling_res=sampling_res, n_traj=v['n_traj'],
                         itr=outer_iter, report=report, limit=v['goal_range'], center=v['goal_center'],
                         bounds=v['goal_range'])
    report.new_row()

    for outer_iter in range(1, v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)
        goals = np.random.uniform(np.array(v['goal_center']) - np.array(v['goal_range']),
                                  np.array(v['goal_center']) + np.array(v['goal_range']), size=(300, v['goal_size']))
        # if needed label the goals before any update
        if v['label_with_variation']:
            old_labels, old_rewards = label_states(goals, env, policy, v['horizon'], as_goals=True, n_traj=v['n_traj'],
                                                   key='goal_reached', full_path=False, return_rew=True)

        with ExperimentLogger(log_dir, 'last', snapshot_mode='last', hold_outter_log=True):
            logger.log("Updating the environment goal generator")
            env.update_goal_generator(
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

            trpo_paths = algo.train()

        if v['use_trpo_paths']:
            logger.log("labeling starts with trpo rollouts")
            [goals, labels] = label_states_from_paths(trpo_paths, n_traj=2, key='goal_reached',  # using the min n_traj
                                                      as_goal=True, env=env)
            paths = [path for paths in trpo_paths for path in paths]
        elif v['label_with_variation']:
            labels, paths = label_states(goals, env, policy, v['horizon'], as_goals=True, n_traj=v['n_traj'],
                                         key='goal_reached', old_rewards=old_rewards, full_path=True)
        else:
            logger.log("labeling starts manually")
            labels, paths = label_states(goals, env, policy, v['horizon'], as_goals=True, n_traj=v['n_traj'],
                                         key='goal_reached', full_path=True)

        with logger.tabular_prefix("OnStarts_"):
            env.log_diagnostics(paths)

        logger.log('Generating the Heatmap...')
        test_and_plot_policy(policy, env, max_reward=v['max_reward'], sampling_res=sampling_res, n_traj=v['n_traj'],
                             itr=outer_iter, report=report, limit=v['goal_range'], center=v['goal_center'],
                             bounds=v['goal_range'])

        # logger.log("Labeling the goals")
        # labels = label_states(goals, env, policy, v['horizon'], n_traj=v['n_traj'], key='goal_reached')

        plot_labeled_states(goals, labels, report=report, itr=outer_iter, limit=v['goal_range'],
                            center=v['goal_center'])

        # ###### extra for deterministic:
        # logger.log("Labeling the goals deterministic")
        # with policy.set_std_to_0():
        #     labels_det = label_states(goals, env, policy, v['horizon'], n_traj=v['n_traj'], n_processes=1)
        # plot_labeled_states(goals, labels_det, report=report, itr=outer_iter, limit=v['goal_range'], center=v['goal_center'])


        logger.dump_tabular(with_prefix=False)
        report.new_row()
