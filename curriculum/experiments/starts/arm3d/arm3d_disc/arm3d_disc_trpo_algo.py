import matplotlib
import cloudpickle
import pickle

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
from curriculum.logging.visualization import save_image, plot_labeled_samples, plot_labeled_states

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab import config

from curriculum.state.evaluator import convert_label, label_states, evaluate_states, evaluate_state_env
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator, \
    StateGenerator
from curriculum.state.utils import StateCollection

from curriculum.envs.start_env import generate_starts
from curriculum.envs.goal_start_env import GoalStartExplorationEnv
from curriculum.envs.maze.maze_evaluate import test_and_plot_policy, sample_unif_feas, unwrap_maze, \
    plot_policy_means
from curriculum.envs.arm3d.arm3d_disc_env import Arm3dDiscEnv

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    logger.log("Initializing report...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=4)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    inner_env = normalize(Arm3dDiscEnv())

    fixed_goal_generator = FixedStateGenerator(state=v['ultimate_goal'])
    fixed_start_generator = FixedStateGenerator(state=v['start_goal'])

    env = GoalStartExplorationEnv(
        env=inner_env,
        start_generator=fixed_start_generator,
        obs2start_transform=lambda x: x[:v['start_size']],
        goal_generator=fixed_goal_generator,
        obs2goal_transform=lambda x: x[-1 * v['goal_size']:],  # the goal are the last 9 coords
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

    if v['baseline'] == 'linear':
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    elif v['baseline'] == 'g_mlp':
        baseline = GaussianMLPBaseline(env_spec=env.spec)

    # load the state collection from data_upload
    load_dir = 'data_upload/state_collections/'
    all_feasible_starts = pickle.load(
        open(osp.join(config.PROJECT_PATH, load_dir, 'disc_all_feasible_states_min.pkl'), 'rb'))
    print("we have %d feasible starts" % all_feasible_starts.size)
    uniform_start_generator = UniformListStateGenerator(state_list=all_feasible_starts.state_list)
    env.update_start_generator(uniform_start_generator)

    logger.log("Training the algorithm")
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v['pg_batch_size'],
        max_path_length=v['horizon'],
        n_itr=v['inner_iters'] * v['outer_iters'],
        step_size=0.01,
        discount=v['discount'],
        plot=False,
    )

    algo.train()

