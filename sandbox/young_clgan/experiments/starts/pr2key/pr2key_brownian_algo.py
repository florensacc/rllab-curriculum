import matplotlib

matplotlib.use('Agg')
import os
import os.path as osp
import random
import time
import numpy as np

from rllab.misc import logger
from collections import OrderedDict
from sandbox.young_clgan.logging import HTMLReport
from sandbox.young_clgan.logging import format_dict
from sandbox.young_clgan.logging.logger import ExperimentLogger
from sandbox.young_clgan.logging.visualization import save_image, plot_labeled_samples, plot_labeled_states

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from sandbox.young_clgan.state.evaluator import convert_label, label_states, evaluate_states, evaluate_state_env
from sandbox.young_clgan.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator, \
    StateGenerator
from sandbox.young_clgan.state.utils import StateCollection

from sandbox.young_clgan.envs.start_env import generate_starts
from sandbox.young_clgan.envs.goal_start_env import GoalStartExplorationEnv
from sandbox.young_clgan.envs.maze.maze_evaluate import test_and_plot_policy, sample_unif_feas, unwrap_maze, \
    plot_policy_means
from sandbox.young_clgan.envs.mjc_key.pr2_key_env import PR2KeyEnv

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    sampling_res = 2 if 'sampling_res' not in v.keys() else v['sampling_res']
    samples_per_cell = 10  # for the oracle rejection sampling

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=4)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    inner_env = normalize(PR2KeyEnv())

    fixed_goal_generator = FixedStateGenerator(state=v['ultimate_goal'])
    uniform_start_generator = UniformStateGenerator(state_size=v['start_size'], bounds=v['start_bounds'])

    env = GoalStartExplorationEnv(
        env=inner_env,
        start_generator=uniform_start_generator,
        obs2start_transform=lambda x: x[:v['start_size']],
        goal_generator=fixed_goal_generator,
        obs2goal_transform=lambda x: x[-1 * v['goal_size']:],  # the goal are the last 9 coords
        terminal_eps=v['terminal_eps'],
        distance_metric=v['distance_metric'],
        extend_dist_rew=v['extend_dist_rew'],
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
# outer_iter = 0
    # logger.log('Generating the Initial Heatmap...')
    # plot_policy_means(policy, env, sampling_res=2, report=report, limit=v['goal_range'], center=v['goal_center'])
    # test_and_plot_policy(policy, env, as_goals=False, max_reward=v['max_reward'], sampling_res=sampling_res,
    #                      n_traj=v['n_traj'],
    #                      itr=outer_iter, report=report, center=v['goal_center'],
    #                      limit=v['goal_range'])  # use goal for plot
    # report.new_row()

    all_starts = StateCollection(distance_threshold=v['coll_eps'])
    seed_starts = generate_starts(env, starts=[v['start_goal']], horizon=v['brownian_horizon'],
                                  variance=v['brownian_variance'], subsample=v['num_new_starts'], animated=True, speedup=0.1)
    # env.update_start_generator(StateGenerator())
    # seed_starts = generate_starts(env, starts=[None], horizon=v['brownian_horizon'],
    #                               variance=v['brownian_variance'], subsample=v['num_new_starts'])

    logger.log("Labeling the seed_starts")
    labels, paths = label_states(seed_starts, env, policy, v['horizon'], as_goals=False, n_traj=v['n_traj'],
                                 key='goal_reached', full_path=True)
    with logger.tabular_prefix("OnStarts_"):
        env.log_diagnostics(paths)

    goal_classes, text_labels = convert_label(labels)
    total_goals = labels.shape[0]
    goal_class_frac = OrderedDict()  # this needs to be an ordered dict!! (for the log tabular)
    for k in text_labels.keys():
        frac = np.sum(goal_classes == k) / total_goals
        logger.record_tabular('GenGoal_frac_' + text_labels[k], frac)
        goal_class_frac[text_labels[k]] = frac



    for outer_iter in range(1, v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)
        logger.log("Sampling starts")

        starts = generate_starts(env, starts=seed_starts, subsample=v['num_new_starts'],
                                 horizon=v['brownian_horizon'], variance=v['brownian_variance'])

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
                step_size=0.01,
                discount=v['discount'],
                plot=False,
            )

            algo.train()

        with logger.tabular_prefix("Uniform_"):
            env.update_start_generator(uniform_start_generator)
            evaluate_state_env(env, policy, horizon=v['horizon'], n_traj=v['n_traj'], n_states=1000)

        logger.log("Labeling the starts")
        labels, paths = label_states(starts, env, policy, v['horizon'], as_goals=False, n_traj=v['n_traj'],
                                     key='goal_reached', full_path=True)
        with logger.tabular_prefix("OnStarts_"):
            env.log_diagnostics(paths)

        goal_classes, text_labels = convert_label(labels)
        total_goals = labels.shape[0]
        goal_class_frac = OrderedDict()  # this needs to be an ordered dict!! (for the log tabular)
        for k in text_labels.keys():
            frac = np.sum(goal_classes == k) / total_goals
            logger.record_tabular('GenGoal_frac_' + text_labels[k], frac)
            goal_class_frac[text_labels[k]] = frac

        # plot_labeled_states(starts, labels, report=report, itr=outer_iter, limit=v['goal_range'],
        #                     center=v['goal_center'], maze_id=v['maze_id'])

        labels = np.logical_and(labels[:, 0], labels[:, 1]).astype(int).reshape((-1, 1))

        logger.dump_tabular(with_prefix=True)
        # report.new_row()

        # append new states to list of all starts (replay buffer): Not the low reward ones!!
        filtered_raw_starts = [start for start, label in zip(starts, labels) if label[0] == 1]
        if len(filtered_raw_starts) > 0:  # add a tone of noise if all the states I had ended up being high_reward!
            seed_starts = filtered_raw_starts
        else:
            seed_starts = generate_starts(env, starts=starts, horizon=v['horizon'] * 2, subsample=v['num_new_starts'],
                                          variance=v['brownian_variance'] * 10)
        all_starts.append(filtered_raw_starts)
