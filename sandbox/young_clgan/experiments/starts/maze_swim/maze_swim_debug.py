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
from sandbox.young_clgan.envs.base import UniformListStateGenerator, UniformStateGenerator, \
    FixedStateGenerator
from sandbox.young_clgan.state.generator import StateGAN
from sandbox.young_clgan.state.utils import StateCollection

from sandbox.young_clgan.envs.goal_env import GoalExplorationEnv, generate_initial_goals
from sandbox.young_clgan.envs.goal_start_env import GoalStartExplorationEnv
from sandbox.young_clgan.envs.maze.maze_evaluate import test_and_plot_policy, plot_policy_means
# from sandbox.young_clgan.envs.maze.maze_ant.ant_maze_env import AntMazeEnv
from rllab.misc.instrument import run_experiment_lite
from rllab.envs.mujoco.maze.swimmer_maze_env import SwimmerMazeEnv

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v = {}):
    v['seed'] = 1
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    sampling_res = 2 if 'sampling_res' not in v.keys() else v['sampling_res']

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    # logger.log("Initializing report and plot_policy_reward...")
    # log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    # report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=3)

    # report.add_header("{}".format(EXPERIMENT_TYPE))
    # report.add_text(format_dict(v))

    v['maze_id'] = 0
    env = normalize(SwimmerMazeEnv(maze_id=v['maze_id']))

    # fixed_goal_generator = FixedStateGenerator(state=v['ultimate_goal'])
    # uniform_start_generator = UniformStateGenerator(state_size=v['start_size'], bounds=v['start_range'],
    #                                                 center=v['start_center'])

    # env = GoalStartExplorationEnv(
    #     env=inner_env,
    #     append_start=v['append_start'],
    #     start_generator=uniform_start_generator,
    #     obs2start_transform=lambda x:  x[:v['start_size']],
    #     goal_generator=fixed_goal_generator,
    #     obs2goal_transform=lambda x:  x[:v['goal_size']],
    #     terminal_eps=v['terminal_eps'],
    #     distance_metric=v['distance_metric'],
    #     extend_dist_rew=v['extend_dist_rew'],
    #     only_feasible=v['only_feasible'],
    #     terminate_env=True,
    # )

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # env_spec=env.spec,
        hidden_sizes=(64, 64),
        # Fix the variance since different goals will require different variances, making this parameter hard to learn.
        # learn_std=v['learn_std'],
        # adaptive_std=v['adaptive_std'],
        # std_hidden_sizes=(16, 16),  # this is only used if adaptive_std is true!
        # output_gain=v['output_gain'],
        # init_std=v['policy_init_std'],
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=2000,
        max_path_length=100,
        n_itr=5000,
        step_size=0.01,
        plot=True,
    )

    algo.train()



run_task()