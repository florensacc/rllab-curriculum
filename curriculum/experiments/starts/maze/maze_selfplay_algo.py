import matplotlib

from curriculum.experiments.asym_selfplay.envs.alice_env import AliceEnv

matplotlib.use('Agg')
import os
import os.path as osp
import random
import numpy as np

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

from curriculum.state.evaluator import convert_label, label_states, evaluate_states, label_states_from_paths
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator
from curriculum.state.utils import StateCollection

from curriculum.envs.start_env import generate_starts, generate_starts_alice
from curriculum.envs.goal_start_env import GoalStartExplorationEnv
from curriculum.envs.maze.maze_evaluate import test_and_plot_policy, sample_unif_feas, unwrap_maze, \
    plot_policy_means
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
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=5)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    inner_env = normalize(PointMazeEnv(maze_id=v['maze_id']))

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
    plot_policy_means(policy, env, sampling_res=sampling_res, report=report, limit=v['goal_range'], center=v['goal_center'])
    test_and_plot_policy(policy, env, as_goals=False, max_reward=v['max_reward'], sampling_res=sampling_res,
                         n_traj=v['n_traj'],
                         itr=outer_iter, report=report, center=v['goal_center'], limit=v['goal_range'])
    report.new_row()

    all_starts = StateCollection(distance_threshold=v['coll_eps'])

    # Use asymmetric self-play to run Alice to generate starts for Bob.
    # Use a double horizon because the horizon is shared between Alice and Bob.
    env_alice = AliceEnv(env_alice=env, env_bob=env, policy_bob=policy, max_path_length=v['alice_horizon'], alice_factor=v['alice_factor'],
                                       alice_bonus=v['alice_bonus'], gamma=1, stop_threshold=v['stop_threshold'])

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

        starts, t_alices = generate_starts_alice(env_alice=env_alice,
                                       algo_alice=algo_alice, start_states=[v['start_goal']],
                                       num_new_starts=v['num_new_starts'], log_dir=log_dir)

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
                step_size=v['step_size'],
                discount=v['discount'],
                plot=False,
            )

            # We don't use these labels anyway, so we might as well take them from training.
            #trpo_paths = algo.train()
            algo.train()

        # logger.log("labeling starts with trpo rollouts")
        # [starts, labels] = label_states_from_paths(trpo_paths, n_traj=2, key='goal_reached',  # using the min n_traj
        #                                            as_goal=False, env=env)
        # paths = [path for paths in trpo_paths for path in paths]

        with logger.tabular_prefix('Outer_'):
            logger.record_tabular('t_alices', np.mean(t_alices))

        logger.log('Generating the Heatmap...')
        plot_policy_means(policy, env, sampling_res=sampling_res, report=report, limit=v['goal_range'], center=v['goal_center'])
        test_and_plot_policy(policy, env, as_goals=False, max_reward=v['max_reward'], sampling_res=sampling_res,
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

        if len(filtered_raw_starts) == 0:  # add a tone of noise if all the states I had ended up being high_reward!
            logger.log("Bad Alice!  All goals are high reward!")

        #     seed_starts = filtered_raw_starts
        # else:
        #     seed_starts = generate_starts(env, starts=starts, horizon=v['horizon'] * 2, subsample=v['num_new_starts'],
        #                                   variance=v['brownian_variance'] * 10)
        all_starts.append(filtered_raw_starts)
