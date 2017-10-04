import matplotlib

from curriculum.algos.sagg_riac.SaggRIAC import SaggRIAC
from curriculum.envs.start_env import generate_starts_alice
from curriculum.experiments.asym_selfplay.envs.alice_env import AliceEnv

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

from curriculum.state.evaluator import label_states, label_states_from_paths, compute_rewards_from_paths
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator
from curriculum.state.generator import StateGAN
from curriculum.state.utils import StateCollection

from curriculum.envs.goal_env import GoalExplorationEnv, generate_initial_goals
from curriculum.envs.maze.maze_evaluate import test_and_plot_policy  # TODO: make this external to maze env
from curriculum.envs.maze.maze_ant.ant_maze_env import AntMazeEnv

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def compute_final_states_from_paths(all_paths, as_goal=True, env=None):
    all_states = []
    for paths in all_paths:
        for path in paths:
            if as_goal:
                state = tuple(env.transform_to_goal_space(path['observations'][-1]))
            else:
                logger.log("Not sure what to do here!!!")
                state = tuple(env.transform_to_start_space(path['observations'][0]))

            all_states.append(state)

    return all_states


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    sampling_res = 2 if 'sampling_res' not in v.keys() else v['sampling_res']

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    if log_dir is None:
        log_dir = "/home/davheld/repos/rllab_goal_rl/data/local/debug"
        debug = True
    else:
        debug = False

    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=5)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    inner_env = normalize(AntMazeEnv(maze_id=v['maze_id']))

    uniform_goal_generator = UniformStateGenerator(state_size=v['goal_size'], bounds=v['goal_range'],
                                                   center=v['goal_center'])
    env = GoalExplorationEnv(
        env=inner_env, goal_generator=uniform_goal_generator,
        obs2goal_transform=lambda x: x[:v['goal_size']],
        terminal_eps=v['terminal_eps'],
        distance_metric=v['distance_metric'],
        extend_dist_rew=v['extend_dist_rew'],
        only_feasible=v['only_feasible'],
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

    outer_iter = 0
    if not debug and not v['fast_mode']:
        logger.log('Generating the Initial Heatmap...')
        test_and_plot_policy(policy, env, max_reward=v['max_reward'], sampling_res=sampling_res, n_traj=v['n_traj'],
                             itr=outer_iter, report=report, limit=v['goal_range'], center=v['goal_center'])

    report.new_row()

    sagg_riac = SaggRIAC(state_size=v['goal_size'],
                         state_range=v['goal_range'],
                         state_center=v['goal_center'],
                         max_goals=v['max_goals'],
                         max_history=v['max_history'])

    for outer_iter in range(1, v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)

        raw_goals = sagg_riac.sample_states(num_samples=v['num_new_goals'])

        goals = raw_goals

        with ExperimentLogger(log_dir, 'last', snapshot_mode='last', hold_outter_log=True):
            logger.log("Updating the environment goal generator")
            env.update_goal_generator(
                UniformListStateGenerator(
                    goals, persistence=v['persistence'], with_replacement=v['with_replacement'],
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

            all_paths = algo.train()

        if v['use_competence_ratio']:
            [goals, rewards] = compute_rewards_from_paths(all_paths, key='competence', as_goal=True, env=env,
                                                          terminal_eps=v['terminal_eps'])
        else:
            [goals, rewards] = compute_rewards_from_paths(all_paths, key='rewards', as_goal=True, env=env)

        [goals_with_labels, labels] = label_states_from_paths(all_paths, n_traj=v['n_traj'], key='goal_reached')
        plot_labeled_states(goals_with_labels, labels, report=report, itr=outer_iter, limit=v['goal_range'],
                            center=v['goal_center'], maze_id=v['maze_id'])

        logger.log('Generating the Heatmap...')
        test_and_plot_policy(policy, env, max_reward=v['max_reward'], sampling_res=sampling_res, n_traj=v['n_traj'],
                             itr=outer_iter, report=report, limit=v['goal_range'], center=v['goal_center'])

        sagg_riac.plot_regions_interest(maze_id=v['maze_id'], report=report)
        sagg_riac.plot_regions_states(maze_id=v['maze_id'], report=report)


        logger.log("Updating SAGG-RIAC")
        sagg_riac.add_states(goals, rewards)

        # Find final states "accidentally" reached by the agent.
        final_goals = compute_final_states_from_paths(all_paths, as_goal=True, env=env)
        sagg_riac.add_accidental_states(final_goals, v['extend_dist_rew'])

        logger.dump_tabular(with_prefix=False)
        report.new_row()

