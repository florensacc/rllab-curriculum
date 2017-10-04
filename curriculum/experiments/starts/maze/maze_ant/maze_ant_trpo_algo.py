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
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from curriculum.state.evaluator import convert_label, label_states, evaluate_states, label_states_from_paths, \
    compute_labels
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator
from curriculum.state.utils import StateCollection

from curriculum.envs.start_env import generate_starts, find_all_feasible_states, check_feasibility, \
    parallel_check_feasibility
from curriculum.envs.goal_start_env import GoalStartExplorationEnv
from curriculum.envs.arm3d.arm3d_disc_env import Arm3dDiscEnv
from curriculum.envs.maze.maze_ant.ant_maze_start_env import AntMazeEnv
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
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=2)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    inner_env = normalize(AntMazeEnv())

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


    if v["baseline"] == "MLP":
        baseline = GaussianMLPBaseline(env_spec=env.spec)
    else:
        baseline = LinearFeatureBaseline(env_spec=env.spec)

    load_dir = 'sandbox/young_clgan/experiments/starts/maze/maze_ant/'
    all_feasible_starts = pickle.load(
        open(osp.join(config.PROJECT_PATH, load_dir, 'good_all_feasible_starts.pkl'), 'rb'))
    logger.log("We have %d feasible starts" % all_feasible_starts.size)

    min_reward = 0.1
    max_reward = 0.9
    improvement_threshold = 0
    old_rewards = None

    uniform_start_generator = UniformListStateGenerator(state_list=all_feasible_starts.state_list)

    init_pos = [[0, 0],
                [1,0],
                 [2,0],
                 [3,0],
                 [4,0],
                 [4,1],
                 [4,2],
                 [4,3],
                [4,4],
                [3,4],
                [2,4],
                [1,4]
                ][::-1]
    for pos in init_pos:
        pos.extend([0.55, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1, ])
    init_pos = np.array(init_pos)

    env.update_start_generator(uniform_start_generator)
    for outer_iter in range(1, v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)
        logger.log("Sampling starts")


        # Following code should be indented
        with ExperimentLogger(log_dir, outer_iter // 50, snapshot_mode='last', hold_outter_log=True):
            logger.log("Updating the environment start generator")
            # env.update_start_generator(uniform_start_generator)
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

        logger.log("Labeling on uniform starts")
        with logger.tabular_prefix("Uniform_"):
            unif_starts = all_feasible_starts.sample(100)
            mean_reward, paths = evaluate_states(unif_starts, env, policy, v['horizon'], n_traj=3, key='goal_reached',
                                                 as_goals=False, full_path=True)
            env.log_diagnostics(paths)
            mean_rewards = mean_reward.reshape(-1, 1)
            labels = compute_labels(mean_rewards, old_rewards=old_rewards, min_reward=min_reward, max_reward=max_reward,
                                    improvement_threshold=improvement_threshold)
            logger.log("Starts labelled")
            plot_labeled_states(unif_starts, labels, report=report, itr=outer_iter, limit=v['goal_range'],
                                center=v['goal_center'], maze_id=v['maze_id'],
                                summary_string_base='initial starts labels:\n')
            report.add_text("Success: " + str(np.mean(mean_reward)))

        with logger.tabular_prefix("Fixed_"):
            mean_reward, paths = evaluate_states(init_pos, env, policy, v['horizon'], n_traj=5, key='goal_reached',
                                                 as_goals=False, full_path=True)
            env.log_diagnostics(paths)
            mean_rewards = mean_reward.reshape(-1, 1)
            labels = compute_labels(mean_rewards, old_rewards=old_rewards, min_reward=min_reward, max_reward=max_reward,
                                    improvement_threshold=improvement_threshold)
            logger.log("Starts labelled")
            plot_labeled_states(init_pos, labels, report=report, itr=outer_iter, limit=v['goal_range'],
                                center=v['goal_center'], maze_id=v['maze_id'],
                                summary_string_base='initial starts labels:\n')
            report.add_text("Fixed Success: " + str(np.mean(mean_reward)))


        report.new_row()
        report.save()
        logger.record_tabular("Fixed test set_success: ", np.mean(mean_reward))
        logger.dump_tabular()

