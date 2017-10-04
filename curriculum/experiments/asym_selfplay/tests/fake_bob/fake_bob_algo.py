import matplotlib

from examples.point_env import PointEnv
from curriculum.experiments.asym_selfplay.envs.alice_env import AliceEnv
from curriculum.experiments.asym_selfplay.envs.alice_fake_env import AliceFakeEnv

matplotlib.use('Agg')
import os
import os.path as osp
import random
import numpy as np

from rllab.misc import logger
from curriculum.logging import HTMLReport
from curriculum.logging import format_dict
from curriculum.experiments.asym_selfplay.tests.fake_bob.visualization import plot_labeled_states

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from curriculum.envs.base import UniformStateGenerator, FixedStateGenerator

from curriculum.envs.start_env import generate_starts_alice
from curriculum.envs.goal_start_env import GoalStartExplorationEnv
from curriculum.envs.maze.point_maze_env import PointMazeEnv

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    sampling_res = 2 if 'sampling_res' not in v.keys() else v['sampling_res']
    samples_per_cell = 10  # for the oracle rejection sampling

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    if log_dir is None:
        log_dir = "/home/davheld/repos/rllab_goal_rl/data/local/debug"
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=4)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    inner_env = normalize(PointMazeEnv(maze_id=v['maze_id'], length=v['maze_length']))
    #inner_env = normalize(PointEnv())

    fixed_goal_generator = FixedStateGenerator(state=v['ultimate_goal'])
    uniform_start_generator = UniformStateGenerator(state_size=v['start_size'], bounds=v['start_range'],
                                                    center=v['start_center'])

    env = GoalStartExplorationEnv(
        env=inner_env,
        start_generator=uniform_start_generator,
        obs2start_transform=lambda x: x[:v['start_size']],
        goal_generator=fixed_goal_generator,
        obs2goal_transform=lambda x: x[:v['goal_size']],
        terminal_eps=v['terminal_eps'],
        distance_metric=v['distance_metric'],
        extend_dist_rew=v['extend_dist_rew'],
        only_feasible=v['only_feasible'],
        terminate_env=True,
    )

    # initialize all logging arrays on itr0
    outer_iter = 0

    # TODO - show initial states for Alice
    report.new_row()

    ring_spacing = 1
    init_iter = 2

    # Use asymmetric self-play to run Alice to generate starts for Bob.
    # Use a double horizon because the horizon is shared between Alice and Bob.
    env_alice = AliceFakeEnv(env, max_path_length=v['alice_horizon'], alice_factor=v['alice_factor'],
                             alice_bonus=v['alice_bonus'], gamma=1, stop_threshold=v['stop_threshold'],
                             ring_spacing=ring_spacing, init_iter=init_iter)

    policy_alice = GaussianMLPPolicy(
            env_spec=env_alice.spec,
            hidden_sizes=(64, 64),
            # Fix the variance since different goals will require different variances, making this parameter hard to learn.
            learn_std=v['learn_std'],
            adaptive_std=v['adaptive_std'],
            std_hidden_sizes=(16, 16),  # this is only used if adaptive_std is true!
            output_gain = v['output_gain_alice'],
            init_std = v['policy_init_std_alice'],
    )
    baseline_alice = LinearFeatureBaseline(env_spec=env_alice.spec)

    algo_alice = TRPO(
        env=env_alice,
        policy=policy_alice,
        baseline=baseline_alice,
        batch_size=v['pg_batch_size_alice'],
        max_path_length=v['alice_horizon'],
        n_itr=v['inner_iters_alice'],
        step_size=0.01,
        discount=v['discount_alice'],
        plot=False,
    )

    for outer_iter in range(1, v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)
        logger.log("Sampling starts")

        # if outer_iter > 10:
        #     init_iter = 5
            #env_alice.set_iter(init_iter)
            #import pdb; pdb.set_trace()

        print("Init iter: " + str(init_iter))

        env_alice = AliceFakeEnv(env, max_path_length=v['alice_horizon'], alice_factor=v['alice_factor'],
                                     alice_bonus=v['alice_bonus'], gamma=1, stop_threshold=v['stop_threshold'],
                                     ring_spacing=ring_spacing, init_iter=init_iter)
        algo_alice.env = env_alice

        #env_alice.set_iter(outer_iter)

        starts, t_alices = generate_starts_alice(env_alice=env_alice, algo_alice=algo_alice,
                                                 start_states=[v['start_goal']], num_new_starts=v['num_new_starts'],
                                                 log_dir=log_dir)

        # Make fake labels
        labels = np.ones([len(starts),2])
        radius = init_iter * ring_spacing
        plot_labeled_states(starts, labels, report=report, itr=outer_iter, limit=v['goal_range'],
                            center=v['goal_center'], maze_id=v['maze_id'],
                            summary_string_base='initial starts labels:\n', radius=radius)
        report.save()

        with logger.tabular_prefix('Outer_'):
            logger.record_tabular('t_alices', np.mean(t_alices))

        logger.dump_tabular(with_prefix=False)
        report.new_row()

