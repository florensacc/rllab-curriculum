import pickle

import matplotlib
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy

# modified to make changes to the environment
from sandbox.ignasi.envs.pr2.pr2_reach_env import Pr2ReachEnv

from sandbox.ignasi.robust_disk.envs.disk_generate_states_env import DiskGenerateStatesEnv

matplotlib.use('Agg')
import os
import os.path as osp
import random
import numpy as np

from rllab.misc import logger
from collections import OrderedDict
from sandbox.young_clgan.logging import HTMLReport
from sandbox.young_clgan.logging import format_dict
from sandbox.young_clgan.logging.logger import ExperimentLogger

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from rllab import config
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy

from sandbox.young_clgan.state.evaluator import convert_label, label_states, evaluate_states, label_states_from_paths
from sandbox.young_clgan.envs.base import UniformListStateGenerator, FixedStateGenerator
from sandbox.young_clgan.state.utils import StateCollection, SmartStateCollection

from sandbox.young_clgan.envs.start_env import generate_starts, find_all_feasible_states
from sandbox.young_clgan.envs.goal_start_env import GoalStartExplorationEnv

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    logger.log("Initializing report...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    if log_dir is None:
        log_dir = "/home/young_clgan"
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=4)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    inner_env = normalize(Pr2ReachEnv(ctrl_regularizer_weight=v['ctrl_regularizer_weight'],
                                      action_torque_lambda=v['action_torque_lambda'],
                                      physics_variances=v['physics_variances'],
                                      disc_mass=v['disc_mass']))

    fixed_goal_generator = FixedStateGenerator(state=v['ultimate_goal'])
    fixed_start_generator = FixedStateGenerator(state=v['start_goal'])

    env = GoalStartExplorationEnv(
        env=inner_env,
        start_generator=fixed_start_generator,
        obs2start_transform=lambda x: x[:v['start_size']], # todo: this is actually wrong as now no joints here!
        goal_generator=fixed_goal_generator,
        obs2goal_transform=lambda x: x[-1 * v['goal_size']:],  # changed!
        terminal_eps=v['terminal_eps'],
        distance_metric=v['distance_metric'],
        extend_dist_rew=v['extend_dist_rew'],
        inner_weight=v['inner_weight'],
        goal_weight=v['goal_weight'],
        terminate_env=True,
        append_goal_to_observation=False,  # prevents goal environment from appending observation
    )

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
            trunc_steps=v['trunc_steps'],
        )
    #
    if v['baseline'] == 'linear':
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    elif v['baseline'] == 'g_mlp':
        baseline = GaussianMLPBaseline(env_spec=env.spec)

    # load the state collection from data_upload
    load_dir = "data_upload/pr2_reach"
    all_feasible_starts = pickle.load(open(osp.join(config.PROJECT_PATH, load_dir, 'all_feasible_states.pkl'), 'rb'))

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

        # with env.set_kill_outside(radius=v['kill_radius']):
        #     seed_starts = generate_starts(env, starts=[v['start_goal']], horizon=500,
        #                                   # animated=True, # speedup=100,  # zero_action=True,
        #                                   variance=1,
        #                                   subsample=100)  # , animated=True, speedup=1)
        #     find_all_feasible_states(env, seed_starts, distance_threshold=0.1,
        #                              brownian_variance=1, max_states=500000,
        #                              horizon=500,
        #                              # animate=True, speedup=100,
        #                              # states_transform= states_transform
        #                              )

        starts = all_feasible_starts.sample(size=v['num_new_starts'])
        s = generate_starts(env, starts=starts.tolist(), horizon=10, variance=1,
                                 animated=True, speedup=1,
                                 zero_action=True,
                                 # states_transform= states_transform
                                 )

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

        logger.log("labeling starts with trpo rollouts")
        [starts, labels] = label_states_from_paths(trpo_paths, n_traj=2, key='goal_reached',
                                                   as_goal=False, env=env)
        paths = [path for paths in trpo_paths for path in paths]

        with logger.tabular_prefix("OnStarts_"):
            env.log_diagnostics(paths)
            algo.sampler.process_samples(itr=outer_iter, paths=trpo_paths[-1])

        start_classes, text_labels = convert_label(labels)
        total_starts = labels.shape[0]
        logger.record_tabular('GenStarts_evaluated', total_starts)
        start_class_frac = OrderedDict()  # this needs to be an ordered dict!! (for the log tabular)
        for k in text_labels.keys():
            frac = np.sum(start_classes == k) / total_starts
            logger.record_tabular('GenStart_frac_' + text_labels[k], frac)
            start_class_frac[text_labels[k]] = frac

        logger.dump_tabular(with_prefix=True)

