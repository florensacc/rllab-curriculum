import matplotlib
from rllab import config
import pickle

from sandbox.young_clgan.algos.sagg_riac.SaggRIAC import SaggRIAC
from sandbox.young_clgan.envs.start_env import generate_starts_alice
from sandbox.young_clgan.experiments.asym_selfplay.envs.alice_env import AliceEnv

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
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy

from sandbox.young_clgan.state.evaluator import convert_label, label_states_from_paths, compute_rewards_from_paths, evaluate_states
from sandbox.young_clgan.envs.base import UniformListStateGenerator, FixedStateGenerator, UniformStateGenerator
from sandbox.young_clgan.state.generator import StateGAN
from sandbox.young_clgan.state.utils import StateCollection

from sandbox.young_clgan.envs.goal_start_env import GoalStartExplorationEnv
from sandbox.ignasi.envs.pr2.pr2_disk_env import Pr2DiskEnv
# from sandbox.ignasi.robust_disk.envs.disk_generate_states_env import DiskGenerateStatesEnv

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def compute_final_states_from_paths(all_paths, as_goal=True, env=None):
    all_states = []
    for paths in all_paths:
        for path in paths:
            if as_goal:
                state = tuple(env.transform_to_goal_space(path['observations'][-1]))
            else:
                logger.log("Not sure what to do here!!!")
                state = tuple(env.transform_to_start_space(path['observations'][0]))

            all_states.append(state)

    return all_states


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    sampling_res = 2 if 'sampling_res' not in v.keys() else v['sampling_res']

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=5)
    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    inner_env = normalize(Pr2DiskEnv())  # todo: actually the env to use should be moving the peg? at least to generate news?

    fixed_goal_generator = FixedStateGenerator(state=v['ultimate_goal'])
    uniform_start_generator = UniformStateGenerator(state_size=v['start_size'], bounds=v['start_range'],  # todo??
                                                   center=v['start_center'])

    env = GoalStartExplorationEnv(
        env=inner_env,
        start_generator=uniform_start_generator,
        obs2start_transform=lambda x: x[:v['start_size']],  # todo: this is actually wrong as now no joints here!
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

    if v['baseline'] == 'linear':
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    elif v['baseline'] == 'g_mlp':
        baseline = GaussianMLPBaseline(env_spec=env.spec)


    bounds = env.observation_space.bounds
    start_bounds = [bounds[0].extend(v['kill_peg_radius'] * np.ones(2)),
                    bounds[1].extend(v['kill_peg_radius'] * np.ones(2))]
    import pdb; pdb.set_trace()
    sagg_riac = SaggRIAC(state_size=v['start_size'],  # todo: change from goals to full task parametrization!!!!
                         state_bounds=start_bounds,
                         max_goals=v['max_goals'],
                         max_history=v['max_history'])

    # load the state collection from data_upload
    load_dir = 'data_upload/pr2_peg'
    all_feasible_starts = pickle.load(open(osp.join(config.PROJECT_PATH, load_dir, 'all_feasible_states_.pkl'), 'rb'))

    for outer_iter in range(1, v['outer_iters']):
        logger.log("Outer itr # %i" % outer_iter)

        starts = sagg_riac.sample_states(num_samples=v['num_new_goals'])

        with ExperimentLogger(log_dir, 'last', snapshot_mode='last', hold_outter_log=True):
            logger.log("Updating the environment goal generator")
            env.update_start_generator(
                UniformListStateGenerator(
                    starts, persistence=v['persistence'], with_replacement=v['with_replacement'],
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

            all_paths = algo.train()

        if v['use_competence_ratio']:
            [goals, rewards] = compute_rewards_from_paths(all_paths, key='competence', as_goal=False, env=env,
                                                          terminal_eps=v['terminal_eps'])
        else:
            [goals, rewards] = compute_rewards_from_paths(all_paths, key='rewards', as_goal=False, env=env)

        [goals_with_labels, labels] = label_states_from_paths(all_paths, n_traj=v['n_traj'], key='goal_reached')

        paths = [path for paths in all_paths for path in paths]
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

        logger.log("Labeling on uniform starts")
        with logger.tabular_prefix("Uniform_"):
            unif_starts = all_feasible_starts.sample(1000)
            mean_reward, paths = evaluate_states(unif_starts, env, policy, v['horizon'], n_traj=1, key='goal_reached',
                                                 as_goals=False, full_path=True)
            env.log_diagnostics(paths)

        logger.dump_tabular(with_prefix=True)

        logger.log("Updating SAGG-RIAC")
        sagg_riac.add_states(goals, rewards)

        # Find final states "accidentally" reached by the agent.
        final_goals = compute_final_states_from_paths(all_paths, as_goal=True, env=env)
        sagg_riac.add_accidental_states(final_goals, v['extend_dist_rew'])

        logger.dump_tabular(with_prefix=False)
        report.new_row()

