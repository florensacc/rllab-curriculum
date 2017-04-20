import matplotlib

matplotlib.use('Agg')
import os
import os.path as osp
import random
import tensorflow as tf
import tflearn

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from sandbox.young_clgan.envs.ndim_point.point_env import PointEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from sandbox.young_clgan.envs.base import GoalIdxExplorationEnv
from sandbox.young_clgan.envs.base import UniformListGoalGenerator, FixedGoalGenerator, UniformGoalGenerator, \
    update_env_goal_generator, generate_onpolicy_goals

from sandbox.young_clgan.goal.evaluator import *
from sandbox.young_clgan.goal.generator import StateGAN
# from sandbox.young_clgan.lib.goal.utils import *
from sandbox.young_clgan.logging.html_report import format_dict, HTMLReport
from sandbox.young_clgan.logging.visualization import *
from sandbox.young_clgan.logging.logger import ExperimentLogger
from sandbox.young_clgan.state.utils import StateCollection

from sandbox.young_clgan.utils import initialize_parallel_sampler

initialize_parallel_sampler()

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    # set loggging
    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=3)
    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    inner_env = normalize(PointEnv(dim=v['goal_size'], state_bounds=v['state_bounds']))

    fixed_goal_generator = FixedGoalGenerator(goal=v['goal_center'])
    feasible_goal_ub = np.array(v['state_bounds'])[:v['goal_size']]
    uniform_feasible_goal_generator = UniformGoalGenerator(goal_size=v['goal_size'], bounds=[-1 * feasible_goal_ub,
                                                                                             feasible_goal_ub])

    env = GoalIdxExplorationEnv(env=inner_env, goal_generator=fixed_goal_generator,
                                idx=np.arange(v['goal_size']),
                                reward_dist_threshold=v['reward_dist_threshold'],
                                distance_metric=v['distance_metric'],
                                dist_goal_weight=v['dist_goal_weight'], max_reward=v['max_reward'],
                                terminal_eps=v['terminal_eps'], terminal_bonus=v['terminal_bonus'],
                                only_feas=v['only_feas'],
                                )  # this goal_generator will be updated by a uniform after

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32),
        # Fix the variance since different goals will require different variances, making this parameter hard to learn.
        learn_std=v['learn_std'],
        adaptive_std=v['adaptive_std'],
        std_hidden_sizes=(16, 16),
        output_gain=v['output_gain'],
        init_std=v['policy_init_std'],
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    all_goals = StateCollection(v['coll_eps'])

    for outer_iter in range(v['outer_iters']):
        logger.log("Outer itr # %i" % outer_iter)

        goals = np.array([]).reshape((-1, v['goal_size']))

        k = 0
        while goals.shape[0] < v['num_new_goals'] and k < 10:
            print('good goals collected: ', goals.shape[0])
            logger.log("Sampling and labeling the goals: %d" % k)
            k += 1
            unif_goals = env.wrapped_env.wrapped_env.observation_space.sample_n(v['num_new_goals'] * 2)[:,
                         :v['goal_size']]  # TODO: fix normal

            labels = label_goals(
                unif_goals, env, policy, v['horizon'],
                min_reward=v['min_reward'],
                max_reward=v['max_reward'],
                old_rewards=None,
                n_traj=v['n_traj'])
            logger.log("Converting the labels")
            init_classes, text_labels = convert_label(labels)
            goals = np.concatenate([goals, unif_goals[init_classes == 2]]).reshape((-1, v['goal_size']))

        if v['replay_buffer'] and all_goals.size > 0:
            old_inits = all_goals.sample(v['num_old_goals'], replay_noise=v['replay_noise'])
            goals = np.vstack([goals, old_inits])

        with ExperimentLogger(log_dir, itr='inner_itr', snapshot_mode='last', hold_outter_log=True):
            # set goal generator to uniformly sample from selected all_goals
            update_env_goal_generator(
                env,
                UniformListGoalGenerator(
                    goals.tolist()
                )
            )

            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=v['pg_batch_size'],
                max_path_length=v['horizon'],
                n_itr=v['inner_iters'],
                discount=v['discount'],  # 0.995
                step_size=0.01,
                plot=False,
                gae_lambda=v['gae_lambda'],
            )

            algo.train()

        # this re-evaluate the final policy in the collection of goals
        logger.log("Generating labels by re-evaluating policy on List of goals...")
        labels = label_goals(
            goals, env, policy, v['horizon'],
            min_reward=v['min_reward'],
            max_reward=v['max_reward'],
            n_traj=v['n_traj'],
        )
        # append new goals to list of all goals (replay buffer): Not the low reward ones!!
        filtered_raw_goals = [goal for goal, label in zip(goals, labels) if label[0] == 1]
        all_goals.append(filtered_raw_goals)

        logger.log("Converting the labels")
        goal_classes, text_labels = convert_label(labels)

        logger.log("Plotting the labeled samples")
        total_goals = labels.shape[0]
        goal_class_frac = OrderedDict()  # this needs to be an ordered dict!! (for the log tabular)
        for k in text_labels.keys():
            frac = np.sum(goal_classes == k) / total_goals
            logger.record_tabular('GenGoal_frac_' + text_labels[k], frac)
            goal_class_frac[text_labels[k]] = frac

        img = plot_labeled_samples(
            samples=goals, sample_classes=goal_classes, text_labels=text_labels, limit=v['goal_range'] + 1,
            bounds=env.feasible_goal_space.bounds,
            # '{}/sampled_goals_{}.png'.format(log_dir, outer_iter),  # if i don't give the file it doesn't save
        )
        summary_string = ''
        for key, value in goal_class_frac.items():
            summary_string += key + ' frac: ' + str(value) + '\n'
        report.add_image(img, 'itr: {}\nLabels of generated goals:\n{}'.format(outer_iter, summary_string), width=500)

        logger.log("Evaluating performance on Unif Goal Gen...")
        with logger.tabular_prefix('UnifFeasGoalGen_'):
            update_env_goal_generator(env, goal_generator=uniform_feasible_goal_generator)
            evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=50, fig_prefix='UnifFeasGoalGen_',
                              report=report, n_traj=v['n_traj'])

        logger.dump_tabular(with_prefix=False)
        report.save()
        report.new_row()

    with logger.tabular_prefix('FINALUnifFeasGoalGen_'):
        update_env_goal_generator(env, goal_generator=uniform_feasible_goal_generator)
        evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=5e3, fig_prefix='FINAL1UnifFeasGoalGen_',
                          report=report, n_traj=v['n_traj'])
        evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=5e3, fig_prefix='FINAL2UnifFeasGoalGen_',
                          report=report, n_traj=v['n_traj'])
    logger.dump_tabular(with_prefix=False)
