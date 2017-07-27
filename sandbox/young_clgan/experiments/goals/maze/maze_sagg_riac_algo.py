import matplotlib

from sandbox.young_clgan.algos.sagg_riac.SaggRIAC import SaggRIAC
from sandbox.young_clgan.envs.start_env import generate_starts_alice
from sandbox.young_clgan.experiments.asym_selfplay.envs.alice_env import AliceEnv

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

from sandbox.young_clgan.state.evaluator import label_states, label_states_from_paths, compute_rewards_from_paths
from sandbox.young_clgan.envs.base import UniformListStateGenerator, UniformStateGenerator
from sandbox.young_clgan.state.generator import StateGAN
from sandbox.young_clgan.state.utils import StateCollection

from sandbox.young_clgan.envs.goal_env import GoalExplorationEnv, generate_initial_goals
from sandbox.young_clgan.envs.maze.maze_evaluate import test_and_plot_policy  # TODO: make this external to maze env
from sandbox.young_clgan.envs.maze.point_maze_env import PointMazeEnv

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
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=3)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    inner_env = normalize(PointMazeEnv(maze_id=v['maze_id']))

    uniform_goal_generator = UniformStateGenerator(state_size=v['goal_size'], bounds=v['goal_range'],
                                                   center=v['goal_center'])
    env = GoalExplorationEnv(
        env=inner_env, goal_generator=uniform_goal_generator,
        #obs2goal_transform=lambda x: x[:int(len(x) / 2)],
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

    if not debug and not v['fast_mode']:
        logger.log('Generating the Initial Heatmap...')
        test_and_plot_policy(policy, env, max_reward=v['max_reward'], sampling_res=sampling_res, n_traj=v['n_traj'],
                             itr=outer_iter, report=report, limit=v['goal_range'], center=v['goal_center'])

    report.new_row()

    all_goals = StateCollection(distance_threshold=v['coll_eps'])

    sagg_riac = SaggRIAC(state_size=v['goal_size'],
                         state_range=v['goal_range'],
                         state_center=v['goal_center'],
                         max_goals=10)

    for outer_iter in range(1, v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)

        raw_goals = sagg_riac.sample_states(num_samples=v['num_new_goals'])

        if v['replay_buffer'] and outer_iter > 0 and all_goals.size > 0:
            old_goals = all_goals.sample(v['num_old_goals'])
            goals = np.vstack([raw_goals, old_goals])
        else:
            goals = raw_goals

        # with ExperimentLogger(log_dir, 'last', snapshot_mode='last', hold_outter_log=True):
        logger.log("Updating the environment goal generator")
        env.update_goal_generator(
            UniformListStateGenerator(
                #goals.tolist(), persistence=v['persistence'], with_replacement=v['with_replacement'],
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

        [goals, rewards] = compute_rewards_from_paths(all_paths, key='goal_reached', as_goal=True, env=env)

        [goals_with_labels, labels] = label_states_from_paths(all_paths, n_traj=v['n_traj'], key='goal_reached')

        # logger.log('Generating the Heatmap...')
        # test_and_plot_policy(policy, env, max_reward=v['max_reward'], sampling_res=sampling_res, n_traj=v['n_traj'],
        #                      itr=outer_iter, report=report, limit=v['goal_range'], center=v['goal_center'])

        #logger.log("Labeling the goals")
        #labels = label_states(goals, env, policy, v['horizon'], n_traj=v['n_traj'], key='goal_reached')

        plot_labeled_states(goals_with_labels, labels, report=report, itr=outer_iter, limit=v['goal_range'],
                           center=v['goal_center'], maze_id=v['maze_id'])

        # ###### extra for deterministic:
        # logger.log("Labeling the goals deterministic")
        # with policy.set_std_to_0():
        #     labels_det = label_states(goals, env, policy, v['horizon'], n_traj=v['n_traj'], n_processes=1)
        # plot_labeled_states(goals, labels_det, report=report, itr=outer_iter, limit=v['goal_range'], center=v['goal_center'])

        #labels = np.logical_and(labels[:, 0], labels[:, 1]).astype(int).reshape((-1, 1))

        logger.log("Updating SAGG-RIAC")
        sagg_riac.add_states(goals, rewards)

        # Find final states "accidentally" reached by the agent.
        final_goals = compute_final_states_from_paths(all_paths, as_goal=True, env=env)
        sagg_riac.add_accidental_states(final_goals)

        logger.dump_tabular(with_prefix=False)
        report.new_row()

        # append new goals to list of all goals (replay buffer): Not the low reward ones!!
        #filtered_raw_goals = [goal for goal, label in zip(goals, labels) if label[0] == 1]
        #all_goals.append(filtered_raw_goals)

        if v['add_on_policy']:
            logger.log("sampling on policy")
            feasible_goals = generate_initial_goals(env, policy, v['goal_range'], goal_center=v['goal_center'],
                                                    horizon=v['horizon'])
            # downsampled_feasible_goals = feasible_goals[np.random.choice(feasible_goals.shape[0], v['add_on_policy']),:]
            all_goals.append(feasible_goals)
