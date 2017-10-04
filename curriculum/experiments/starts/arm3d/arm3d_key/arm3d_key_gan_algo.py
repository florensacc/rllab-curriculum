import matplotlib
import cloudpickle
import pickle

matplotlib.use('Agg')
import os
import os.path as osp
import random
import time
import numpy as np
import tensorflow as tf
import tflearn

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
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from curriculum.state.evaluator import convert_label, label_states, evaluate_states, label_states_from_paths
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator, \
    StateGenerator
from curriculum.state.utils import StateCollection
from curriculum.state.generator import StateGAN

from curriculum.envs.start_env import generate_starts
from curriculum.envs.goal_start_env import GoalStartExplorationEnv
from curriculum.envs.arm3d.arm3d_key_env import Arm3dKeyEnv

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

    tf_session = tf.Session()

    inner_env = normalize(Arm3dKeyEnv(ctrl_cost_coeff=v['ctrl_cost_coeff']))

    fixed_goal_generator = FixedStateGenerator(state=v['ultimate_goal'])
    fixed_start_generator = FixedStateGenerator(state=v['ultimate_goal'])

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

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    # load the state collection from data_upload
    load_dir = 'data_upload/state_collections/'
    all_feasible_starts = pickle.load(open(osp.join(config.PROJECT_PATH, load_dir, 'key_all_feasible_states.pkl'), 'rb'))
    print("we have %d feasible starts" % all_feasible_starts.size)

    # GAN
    logger.log("Instantiating the GAN...")
    gan_configs = {key[4:]: value for key, value in v.items() if 'GAN_' in key}
    for key, value in gan_configs.items():
        if value is tf.train.AdamOptimizer:
            gan_configs[key] = tf.train.AdamOptimizer(gan_configs[key + '_stepSize'])
        if value is tflearn.initializations.truncated_normal:
            gan_configs[key] = tflearn.initializations.truncated_normal(stddev=gan_configs[key + '_stddev'])

    gan = StateGAN(
        state_size=v['start_size'],
        evaluater_size=v['num_labels'],
        state_bounds=v['start_bounds'],
        state_noise_level=v['start_noise_level'],
        generator_layers=v['gan_generator_layers'],
        discriminator_layers=v['gan_discriminator_layers'],
        noise_size=v['gan_noise_size'],
        tf_session=tf_session,
        configs=gan_configs,
    )
    logger.log("pretraining the GAN...")
    if v['smart_init']:
        feasible_starts = generate_starts(env, starts=[v['start_goal']], horizon=50, size=100000)  # , animated=True, speedup=10)  # without giving the policy it does brownian mo.
        dis_loss, gen_loss = gan.pretrain(states=feasible_starts, outer_iters=v['gan_outer_iters'])
        print("Loss of Gen and Dis: ", gen_loss, dis_loss)
    else:
        gan.pretrain_uniform(outer_iters=500, report=report)  # v['gan_outer_iters'])

    all_starts = StateCollection(distance_threshold=v['coll_eps'])

    # # show where these states are:
    # print("showing the states the GAN trained on:")
    # shuffled_starts = np.array(feasible_starts)
    # np.random.shuffle(shuffled_starts)
    # generate_starts(env, starts=shuffled_starts, horizon=100, zero_action=True, animated=True, speedup=10)

    for outer_iter in range(1, v['outer_iters']):


        logger.log("Outer itr # %i" % outer_iter)
        logger.log("Sampling starts")
        raw_starts, _ = gan.sample_states_with_noise(v['num_new_starts'])

        # show where these states are:
        shuffled_starts = np.array(raw_starts)
        np.random.shuffle(shuffled_starts)
        generate_starts(env, starts=shuffled_starts, horizon=100, zero_action=True, animated=True, speedup=10)

        if v['replay_buffer'] and outer_iter > 0 and all_starts.size > 0:
            old_starts = all_starts.sample(v['num_old_starts'])
            starts = np.vstack([starts, old_starts])
        else:
            starts = raw_starts

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

            trpo_paths = algo.train()

        if v['use_trpo_paths']:
            logger.log("labeling starts with trpo rollouts")
            [starts, labels] = label_states_from_paths(trpo_paths, n_traj=2, key='goal_reached',  # using the min n_traj
                                                                 as_goal=False, env=env)
            paths = [path for paths in trpo_paths for path in paths]
        else:
            logger.log("labeling starts manually")
            labels, paths = label_states(starts, env, policy, v['horizon'], as_goals=False, n_traj=v['n_traj'],
                                         key='goal_reached', full_path=True)

        with logger.tabular_prefix("OnStarts_"):
            env.log_diagnostics(paths)

        start_classes, text_labels = convert_label(labels)
        total_starts = labels.shape[0]
        logger.record_tabular('GenStarts_evaluated', total_starts)
        start_class_frac = OrderedDict()  # this needs to be an ordered dict!! (for the log tabular)
        for k in text_labels.keys():
            frac = np.sum(start_classes == k) / total_starts
            logger.record_tabular('GenStart_frac_' + text_labels[k], frac)
            start_class_frac[text_labels[k]] = frac

        labels = np.logical_and(labels[:, 0], labels[:, 1]).astype(int).reshape((-1, 1))

        logger.log("Training the GAN")
        if np.any(labels):
            gan.train(
                starts, labels,
                v['gan_outer_iters'],
            )

        logger.log("Labeling on uniform starts")
        with logger.tabular_prefix("Uniform_"):
            unif_starts = all_feasible_starts.sample(1000)
            mean_reward, paths = evaluate_states(unif_starts, env, policy, v['horizon'], n_traj=1, key='goal_reached',
                                                 as_goals=False, full_path=True)
            env.log_diagnostics(paths)

        logger.dump_tabular(with_prefix=True)

        # append new states to list of all starts (replay buffer): Not the low reward ones!!
        logger.log("Appending good goals to replay and generating seeds")
        filtered_raw_starts = [start for start, label in zip(starts, labels) if label[0] == 1]
        all_starts.append(filtered_raw_starts)

        # if v['seed_with'] == 'only_goods':
        #     if len(filtered_raw_starts) > 0:  # add a tone of noise if all the states I had ended up being high_reward!
        #         seed_starts = filtered_raw_starts
        #     elif np.sum(start_classes == 0) > np.sum(start_classes == 1):  # if more low reward than high reward
        #         seed_starts = all_starts.sample(300)  # sample them from the replay
        #     else:
        #         with env.set_kill_outside():
        #             seed_starts = generate_starts(env, starts=starts, horizon=int(v['horizon'] * 10), subsample=v['num_new_starts'],
        #                                           variance=v['brownian_variance'] * 10)
        # elif v['seed_with'] == 'all_previous':
        #     seed_starts = starts
        # elif v['seed_with'] == 'on_policy':
        #     with env.set_kill_outside():
        #         seed_starts = generate_starts(env, policy, horizon=v['horizon'], subsample=v['num_new_starts'])
