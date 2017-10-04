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
from curriculum.state.utils import StateCollection, SmartStateCollection

from curriculum.envs.start_env import generate_starts, find_all_feasible_states, check_feasibility, \
    parallel_check_feasibility
from curriculum.envs.goal_start_env import GoalStartExplorationEnv
from curriculum.envs.arm3d.arm3d_disc_env import Arm3dDiscEnv
from curriculum.envs.maze.maze_ant.ant_maze_start_env import AntMazeEnv
from curriculum.envs.maze.maze_evaluate import test_and_plot_policy, sample_unif_feas, unwrap_maze, \
    plot_policy_means
import pickle


EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    logger.log("Initializing report...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    if log_dir is None:
        log_dir = "/home/michael/"
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=3)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    inner_env = normalize(AntMazeEnv())

    fixed_goal_generator = FixedStateGenerator(state=v['ultimate_goal'])
    fixed_start_generator = FixedStateGenerator(state=v['ultimate_goal'])


    load_dir = 'sandbox/young_clgan/experiments/starts/maze/maze_ant/'
    save_dir = 'data/debug/'
    # with open(os.path.join(config.PROJECT_PATH, save_dir, "test.pkl"), 'wb') as handle:
    #     pickle.dump({}, handle)

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

    #baseline = LinearFeatureBaseline(env_spec=env.spec)
    if v["baseline"] == "MLP":
        baseline = GaussianMLPBaseline(env_spec=env.spec)
    else:
        baseline = LinearFeatureBaseline(env_spec=env.spec)

    # load the state collection from data_upload

    if v['smart_replay_buffer']:
        all_starts = SmartStateCollection(distance_threshold=v['coll_eps'], states_transform=lambda x: x[:, :2],
                                          abs=v["smart_replay_abs"],
                                          eps=v["smart_replay_eps"]
                                          )
    else:
        all_starts = StateCollection(distance_threshold=v['coll_eps'], states_transform=lambda x: x[:, :2])
    # brownian_starts = StateCollection(distance_threshold=v['regularize_starts'])
    # with env.set_kill_outside():
    # initial brownian horizon and size are pretty important
    logger.log("Brownian horizon: {}".format(v['initial_brownian_horizon']))
    seed_starts = generate_starts(env, starts=[v['start_goal']], horizon=v['initial_brownian_horizon'], size=15000, # this is smaller as they are seeds!
                                  variance=v['brownian_variance'],
                                  animated=False,
                                  # subsample=v['num_new_starts'],
                                  )  # , animated=True, speedup=1)

    if v['filter_bad_starts']:
        logger.log("Prefilter seed starts: {}".format(len(seed_starts)))
        seed_starts = parallel_check_feasibility(env=env, starts=seed_starts, max_path_length=v['feasibility_path_length'])
        logger.log("Filtered seed starts: {}".format(len(seed_starts)))

    # can also filter these starts optionally


    all_feasible_starts = pickle.load(
        open(osp.join(config.PROJECT_PATH, load_dir, 'good_all_feasible_starts.pkl'), 'rb'))
    logger.log("We have %d feasible starts" % all_feasible_starts.size)

    min_reward = 0.1
    max_reward = 0.9
    improvement_threshold = 0
    old_rewards = None

    init_pos = [[0, 0],
                [1, 0],
                [2, 0],
                [3, 0],
                [4, 0],
                [4, 1],
                [4, 2],
                [4, 3],
                [4, 4],
                [3, 4],
                [2, 4],
                [1, 4]
                ][::-1]
    for pos in init_pos:
        pos.extend([0.55, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1, ])
    init_pos = np.array(init_pos)


    for outer_iter in range(1, v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)
        logger.log("Sampling starts")

        report.save()

        # generate starts from the previous seed starts, which are defined below
        starts = generate_starts(env, starts=seed_starts, subsample=v['num_new_starts'], size=2000,
                                 horizon=v['brownian_horizon'], variance=v['brownian_variance'])

        # note: this messes with the balance between starts and old_starts!
        if v['filter_bad_starts']:
            logger.log("Prefilter starts: {}".format(len(starts)))
            starts = parallel_check_feasibility(env=env, starts=starts, max_path_length=v['feasibility_path_length'])
            logger.log("Filtered starts: {}".format(len(starts)))

        logger.log("Total number of starts in buffer: {}".format(all_starts.size))
        if v['replay_buffer'] and outer_iter > 0 and all_starts.size > 0:
            # with open(os.path.join(config.PROJECT_PATH, save_dir, "qval{}.pkl".format(outer_iter)), 'wb') as handle:
            #     pickle.dump(all_starts.q_vals, handle)
            # with open(os.path.join(config.PROJECT_PATH, save_dir, "preval{}.pkl".format(outer_iter)), 'wb') as handle:
            #     pickle.dump(all_starts.prev_vals, handle)
            old_starts = all_starts.sample(v['num_old_starts'])
            starts = np.vstack([starts, old_starts])

        # plot starts before training
        # takes too much time
        # labels = label_states(starts, env, policy, v['horizon'],
        #                       as_goals=False, n_traj=v['n_traj'], key='goal_reached')
        # plot_labeled_states(starts, labels, report=report, itr=outer_iter, limit=v['goal_range'],
        #                     center=v['goal_center'], maze_id=v['maze_id'],
        #                     summary_string_base='initial starts labels:\n')


        # Following code should be indented
        with ExperimentLogger(log_dir, outer_iter // 50, snapshot_mode='last', hold_outter_log=True):
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

            trpo_paths = algo.train()



        logger.log("Labeling the starts")

        if v['smart_replay_buffer']:
            [starts, labels, mean_rewards] = label_states_from_paths(trpo_paths, n_traj=v['n_traj'], key='goal_reached',
                                                   as_goal=False, env=env, return_mean_rewards=True)
        else:
            [starts, labels] = label_states_from_paths(trpo_paths, n_traj=v['n_traj'], key='goal_reached',
                                                       # using the min n_traj
                                                       as_goal=False, env=env)

        start_classes, text_labels = convert_label(labels)
        plot_labeled_states(starts, labels, report=report, itr=outer_iter, limit=v['goal_range'],
                            center=v['goal_center'], maze_id=v['maze_id'])


        labels = np.logical_and(labels[:, 0], labels[:, 1]).astype(int).reshape((-1, 1))

        # append new states to list of all starts (replay buffer): Not the low reward ones!!


        if v['seed_with'] == 'only_goods':
            filtered_raw_starts = [start for start, label in zip(starts, labels) if label[0] == 1]
            if len(filtered_raw_starts) > 0:  # add a ton of noise if all the states I had ended up being high_reward!
                logger.log("We have {} good starts!".format(len(filtered_raw_starts)))
                seed_starts = filtered_raw_starts
            elif np.sum(start_classes == 0) > np.sum(start_classes == 1):  # if more low reward than high reward
                logger.log("More bad starts than good starts, sampling seeds from replay buffer")
                seed_starts = all_starts.sample(300)  # sample them from the replay
            else:
                logger.log("More good starts than bad starts, resampling")
                seed_starts = generate_starts(env, starts=starts, horizon=v['horizon'] * 2, subsample=v['num_new_starts'], size=10000,
                                              variance=v['brownian_variance'] * 10)

        elif v['seed_with'] == 'all_previous':
            seed_starts = starts
            filtered_raw_starts = starts # no filtering done
        else:
            raise Exception

        # update replay buffer!
        if v['smart_replay_buffer']:
            # within the replay buffer, we can choose to disregard states that have a reward between 0 and 1
            if v['seed_with'] == 'only_goods':
                logger.log("Only goods and smart replay buffer (probably best option)")
                all_starts.update_starts(starts, mean_rewards, True, logger)
            else:
                all_starts.update_starts(starts, mean_rewards, False, logger)
        elif v['seed_with'] == 'only_goods' or v['seed_with'] == 'all_previous':
            all_starts.append(filtered_raw_starts)
        else:
            raise Exception



        # need to put this last! otherwise labels variable gets confused
        logger.log("Labeling on uniform starts")
        if not v["debug"]:
            with logger.tabular_prefix("Uniform_"):
                unif_starts = all_feasible_starts.sample(100)
                mean_reward, paths = evaluate_states(unif_starts, env, policy, v['horizon'], n_traj=v['n_traj'], key='goal_reached',
                                                     as_goals=False, full_path=True)
                env.log_diagnostics(paths)
                mean_rewards = mean_reward.reshape(-1, 1)
                labels = compute_labels(mean_rewards, old_rewards=old_rewards, min_reward=min_reward, max_reward=max_reward,
                                        improvement_threshold=improvement_threshold)
                logger.log("Starts labelled")
                plot_labeled_states(unif_starts, labels, report=report, itr=outer_iter, limit=v['goal_range'],
                                    center=v['goal_center'], maze_id=v['maze_id'],
                                    summary_string_base='initial starts labels:\n')
                # report.add_text("Success: " + str(np.mean(mean_reward)))

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
