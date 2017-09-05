import pickle

import matplotlib
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy

# modified to make changes to the environment
from sandbox.ignasi.envs.pr2.pr2_disk import Pr2DiskEnv

from sandbox.ignasi.robust_disk.envs.disk_generate_states_env import DiskGenerateStatesEnv

matplotlib.use('Agg')
import os
import os.path as osp
import random
import numpy as np

from rllab.misc import logger
from collections import OrderedDict
from sandbox.ignasi.logging import HTMLReport
from sandbox.ignasi.logging import format_dict
from sandbox.ignasi.logging.logger import ExperimentLogger

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from rllab import config
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy

from sandbox.ignasi.state.evaluator import convert_label, label_states, evaluate_states, label_states_from_paths
from sandbox.ignasi.envs.base import UniformListStateGenerator, FixedStateGenerator
from sandbox.ignasi.state.utils import StateCollection, SmartStateCollection

from sandbox.ignasi.envs.start_env import generate_starts, find_all_feasible_states
from sandbox.ignasi.envs.goal_start_env import GoalStartExplorationEnv

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    logger.log("Initializing report...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    if log_dir is None:
        log_dir = "/home/ignasi"
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=4)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    inner_env = normalize(Pr2DiskEnv())

    fixed_goal_generator = FixedStateGenerator(state=v['ultimate_goal'])
    fixed_start_generator = FixedStateGenerator(state=v['ultimate_goal'])

    env = GoalStartExplorationEnv(
        env=inner_env,
        start_generator=fixed_start_generator,
        obs2start_transform=lambda x: x[:v['start_size']],
        goal_generator=fixed_goal_generator,
        obs2goal_transform=lambda x: x[-1 * v['goal_size']:], # changed!
        terminal_eps=v['terminal_eps'],
        distance_metric=v['distance_metric'],
        extend_dist_rew=v['extend_dist_rew'],
        inner_weight=v['inner_weight'],
        goal_weight=v['goal_weight'],
        terminate_env=True,
        append_goal_to_observation = False, # prevents goal environment from appending observation
    )

    if v['move_peg']:  # todo: change this to the PR2 model
        gen_states_env = DiskGenerateStatesEnv(kill_peg_radius=v['kill_peg_radius'], kill_radius=v['kill_radius'])
    else:
        # cannot move the peg
        gen_states_env = env

    if v['policy'] == 'mlp':
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
    elif v['policy'] == 'recurrent':
        policy = GaussianGRUPolicy(
            env_spec=env.spec,
            hidden_sizes=(32,),
            learn_std=v['learn_std'],
        )
    #
    if v['baseline'] == 'linear':
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    elif v['baseline'] == 'g_mlp':
        baseline = GaussianMLPBaseline(env_spec=env.spec)

    # load the state collection from data_upload
    load_dir = 'data_upload/peg'
    all_feasible_starts = pickle.load(open(osp.join(config.PROJECT_PATH, load_dir, 'all_feasible_states.pkl'), 'rb'))
    print("we have %d feasible starts" % all_feasible_starts.size)


    if v['smart_replay_buffer']:
        all_starts = SmartStateCollection(distance_threshold=v['coll_eps'],
                                          abs=v["smart_replay_abs"],
                                          eps=v["smart_replay_eps"]
                                          )
    else:
        all_starts = StateCollection(distance_threshold=v['coll_eps'])
    brownian_starts = StateCollection(distance_threshold=v['regularize_starts'])
    with gen_states_env.set_kill_outside():
        seed_starts = generate_starts(gen_states_env, starts=[v['start_goal']], horizon=100 * v['brownian_horizon'], animated=True, speedup=100,
                                      variance=v['brownian_variance'], subsample=v['num_new_starts'], zero_action=True
                                      )
                                      #   animated=True, speedup=1)

    if v['generating_test_set']:
        # change distance metric for states generated
        if v['peg_positions']:
            def states_transform(x):
                y = list(x)
                for joint in (7,8):
                    y[joint] *= 10
                # for joint in v['peg_positions']:
                #     y[joint] *= v['peg_scaling']
                return y
        else:
            states_transform = None
        with gen_states_env.set_kill_outside():
            find_all_feasible_states(gen_states_env, seed_starts, distance_threshold=0.1,
                                     brownian_variance=1, animate=False, max_states=v['max_gen_states'],
                                     horizon=500,
                                     states_transform= states_transform
                                     )
            import sys; sys.exit(0)

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

    for outer_iter in range(1, v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)
        logger.log("Sampling starts")

        all_starts.states.dump(osp.join(log_dir, 'buffer_states.pkl'))
        # all_starts.states.dump("/home/michael/rllab_goal_rl/data/check_buffer/buffer_states.pkl")

        with gen_states_env.set_kill_outside():
            starts = generate_starts(gen_states_env, starts=seed_starts, horizon=v['brownian_horizon'],
                                     variance=v['brownian_variance'],
                                     animated=False,
                                     )

        # regularization of the brownian starts
        brownian_starts.empty()
        brownian_starts.append(starts)
        starts = brownian_starts.sample(size=v['num_new_starts'])

        if v['replay_buffer'] and outer_iter > 0 and all_starts.size > 0:
            # can squeeze here
            old_starts = all_starts.sample(v['num_old_starts'])
            starts = np.vstack([starts, old_starts])

        # todo: indent!!
        with ExperimentLogger(log_dir, outer_iter // 10, snapshot_mode='last', hold_outter_log=True):
            logger.log("Updating the environment start generator")
            env.update_start_generator(
                UniformListStateGenerator(
                    starts.tolist(), persistence=v['persistence'], with_replacement=v['with_replacement'],
                )
            )

            logger.log("Training the algorithm")

            algo.current_itr = 0
            trpo_paths = algo.train(already_init=outer_iter > 1)


        if v['use_trpo_paths']:
            logger.log("labeling starts with trpo rollouts")
            if v['smart_replay_buffer']:
                [starts, labels, mean_rewards] = label_states_from_paths(trpo_paths, n_traj=v['n_traj'],
                                                                         key='goal_reached',
                                                                         as_goal=False, env=env,
                                                                         return_mean_rewards=True)
            else:
                [starts, labels] = label_states_from_paths(trpo_paths, n_traj=2, key='goal_reached',  # using the min n_traj
                                                                 as_goal=False, env=env)
            paths = [path for paths in trpo_paths for path in paths]
        else:
            logger.log("labeling starts manually")
            labels, paths = label_states(starts, env, policy, v['horizon'], as_goals=False, n_traj=v['n_traj'],
                                         key='goal_reached', full_path=True)

        with logger.tabular_prefix("OnStarts_"):
            env.log_diagnostics(paths)
        logger.record_tabular('brownian_starts', brownian_starts.size)

        start_classes, text_labels = convert_label(labels)
        total_starts = labels.shape[0]
        logger.record_tabular('GenStarts_evaluated', total_starts)
        start_class_frac = OrderedDict()  # this needs to be an ordered dict!! (for the log tabular)
        for k in text_labels.keys():
            frac = np.sum(start_classes == k) / total_starts
            logger.record_tabular('GenStart_frac_' + text_labels[k], frac)
            start_class_frac[text_labels[k]] = frac

        labels = np.logical_and(labels[:, 0], labels[:, 1]).astype(int).reshape((-1, 1))

        logger.log("Labeling on uniform starts")
        with logger.tabular_prefix("Uniform_"):
            unif_starts = all_feasible_starts.sample(1000)
            mean_reward, paths = evaluate_states(unif_starts, env, policy, v['horizon'], n_traj=1, key='goal_reached',
                                                 as_goals=False, full_path=True)
            env.log_diagnostics(paths)

        logger.dump_tabular(with_prefix=True)

        # append new states to list of all starts (replay buffer): Not the low reward ones!!
        logger.log("Appending good goals to replay and generating seeds")
        logger.log("Number of raw starts")
        filtered_raw_starts = [start for start, label in zip(starts, labels) if label[0] == 1]


        if v['seed_with'] == 'only_goods':
            if len(filtered_raw_starts) > 0:  # add a tone of noise if all the states I had ended up being high_reward!
                seed_starts = filtered_raw_starts
            elif np.sum(start_classes == 0) > np.sum(start_classes == 1):  # if more low reward than high reward
                seed_starts = all_starts.sample(300)  # sample them from the replay
            else:
                with gen_states_env.set_kill_outside():
                    seed_starts = generate_starts(gen_states_env, starts=starts, horizon=int(v['horizon'] * 10), subsample=v['num_new_starts'],
                                                  variance=v['brownian_variance'] * 10)
        elif v['seed_with'] == 'all_previous':
            seed_starts = starts
        elif v['seed_with'] == 'on_policy':
            with gen_states_env.set_kill_outside():
                seed_starts = generate_starts(gen_states_env, policy, horizon=v['horizon'], subsample=v['num_new_starts'])


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
