import matplotlib
matplotlib.use('Agg')
import os
import os.path as osp
import random
import numpy as np
from collections import OrderedDict

from sandbox.young_clgan.envs.maze.maze_evaluate_old import test_and_plot_policy  # TODO: unify interface w/ Init
from sandbox.young_clgan.envs.maze.point_maze_env import PointMazeEnv
from sandbox.young_clgan.logging import HTMLReport
from sandbox.young_clgan.logging import format_dict
from sandbox.young_clgan.logging.visualization import save_image, plot_labeled_samples, \
    plot_line_graph

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from sandbox.young_clgan.envs.base import UniformListGoalGenerator, FixedGoalGenerator, update_env_goal_generator, \
    generate_onpolicy_goals

from sandbox.young_clgan.logging.logger import ExperimentLogger
from sandbox.young_clgan.goal.evaluator import convert_label, label_goals, evaluate_goals
from sandbox.young_clgan.state.utils import StateCollection

from rllab.misc import logger

from sandbox.young_clgan.utils import initialize_parallel_sampler

initialize_parallel_sampler()

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=3)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    env = normalize(PointMazeEnv(
        goal_generator=FixedGoalGenerator([0.1, 0.1]),
        reward_dist_threshold=v['reward_dist_threshold'],
        # reward_dist_threshold=v['reward_dist_threshold'] * 0.1,
        # append_goal=False,
        indicator_reward=v['indicator_reward'],
        terminal_eps=v['terminal_eps'],
        only_feas=v['only_feas'],
    ))

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
    all_mean_rewards = []
    all_success = []
    outer_iter = 0
    n_traj = 3 if v['indicator_reward'] else 1
    sampling_res = 2
    # logger.log('Generating the Initial Heatmap...')
    # avg_rewards, avg_success, heatmap = test_and_plot_policy(policy, env, max_reward=v['max_reward'],
    #                                                          sampling_res=sampling_res, n_traj=n_traj)
    # reward_img = save_image()

    # mean_rewards = np.mean(avg_rewards)
    # success = np.mean(avg_success)

    # all_mean_rewards.append(mean_rewards)
    # all_success.append(success)
    #
    # with logger.tabular_prefix('Outer_'):
    #     logger.record_tabular('iter', outer_iter)
    #     logger.record_tabular('MeanRewards', mean_rewards)
    #     logger.record_tabular('Success', success)
    # # logger.dump_tabular(with_prefix=False)
    #
    # report.add_image(
    #     reward_img,
    #     'policy performance\n itr: {} \nmean_rewards: {} \nsuccess: {}'.format(
    #         outer_iter, all_mean_rewards[-1],
    #         all_success[-1]
    #     )
    # )

    on_policy_goals = generate_onpolicy_goals(env, policy, goal_range=v['goal_range'],
                                              center=v['goal_center'], horizon=v['horizon'], size=v['onpolicy_samples'])
    img = plot_labeled_samples(on_policy_goals, sample_classes=np.zeros(len(on_policy_goals), dtype=int),
                               colors=['k'], text_labels={0: 'training states'}, limit=v['goal_range'],
                               center=v['goal_center'])
    report.add_image(img, 'states used as goals next time', width=500)

    indices = np.random.randint(0, on_policy_goals.shape[0], v['num_goals'])
    initial_goals = on_policy_goals[indices, :]
    initial_goals += v['goal_noise'] * np.random.randn(*initial_goals.shape)
    logger.log("Labeling the goals")
    labels = label_goals(
        initial_goals, env, policy, v['horizon'],
        min_reward=v['min_reward'],
        max_reward=v['max_reward'],
        old_rewards=None,
        improvement_threshold=v['improvement_threshold'],
        n_traj=n_traj)

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
        samples=initial_goals, sample_classes=goal_classes, text_labels=text_labels, limit=v['goal_range'],
        center=v['goal_center']
        # '{}/sampled_goals_{}.png'.format(log_dir, outer_iter),  # if i don't give the file it doesn't save
    )
    summary_string = ''
    for key, value in goal_class_frac.items():
        summary_string += key + ' frac: ' + str(value) + '\n'
    report.add_image(img, 'itr: {}\nLabels of generated goals:\n{}'.format(outer_iter, summary_string), width=500)
    report.save()
    report.new_row()

    for outer_iter in range(1, v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)
        indices = np.random.randint(0, on_policy_goals.shape[0], v['num_goals'])
        goals = on_policy_goals[indices, :]
        goals += v['goal_noise'] * np.random.randn(*goals.shape)

        with ExperimentLogger(log_dir, snapshot_mode='last', hold_outter_log=True):
            logger.log("Updating the environment goal generator")
            update_env_goal_generator(
                env,
                UniformListGoalGenerator(
                    goals.tolist()
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
                discount=v['discount'],
                step_size=0.01,
                plot=False,
                gae_lambda=v['gae_lambda'])

            algo.train()

        logger.log('Generating the Heatmap...')
        avg_rewards, avg_success, heatmap = test_and_plot_policy(policy, env,
                                                                 # max_reward=v['max_reward'],  #max_rew just for 300
                                                                 sampling_res=sampling_res, n_traj=n_traj,
                                                                 visualize=False)
        reward_img = save_image()

        mean_rewards = np.mean(avg_rewards)
        success = np.mean(avg_success)

        all_mean_rewards.append(mean_rewards)
        all_success.append(success)

        with logger.tabular_prefix('Outer_'):
            logger.record_tabular('iter', outer_iter)
            logger.record_tabular('MeanRewards', mean_rewards)
            logger.record_tabular('Success', success)

        report.add_image(
            reward_img,
            'policy performance\n itr: {} \nmean_rewards: {}\nsuccess: {}'.format(
                outer_iter, all_mean_rewards[-1],
                all_success[-1]
            )
        )
        report.save()

        logger.log("Labeling the goals")
        labels = label_goals(
            goals, env, policy, v['horizon'],
            min_reward=v['min_reward'],
            max_reward=v['max_reward'],
            improvement_threshold=v['improvement_threshold'],
            n_traj=n_traj)

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
            samples=goals, sample_classes=goal_classes, text_labels=text_labels,
            limit=v['goal_range'], center=v['goal_center'],
            # '{}/sampled_goals_{}.png'.format(log_dir, outer_iter),  # if i don't give the file it doesn't save
        )
        summary_string = ''
        for key, value in goal_class_frac.items():
            summary_string += key + ' frac: ' + str(value) + '\n'
        report.add_image(img, 'itr: {}\nLabels of generated goals:\n{}'.format(outer_iter, summary_string), width=500)

        on_policy_goals = generate_onpolicy_goals(env, policy, v['goal_range'], center=v['goal_center'],
                                                  horizon=v['horizon'], size=v['onpolicy_samples'])

        img = plot_labeled_samples(on_policy_goals, sample_classes=np.zeros(len(on_policy_goals), dtype=int),
                                   colors=['k'], text_labels={0: 'training states'}, limit=v['goal_range'],
                                   center=v['goal_center'])
        report.add_image(img, 'states used as goals for NEXT itr', width=500)

        logger.dump_tabular(with_prefix=False)
        report.save()
        report.new_row()

    img = plot_line_graph(
        osp.join(log_dir, 'mean_rewards.png'),
        range(v['outer_iters']), all_mean_rewards
    )
    report.add_image(img, 'Mean rewards', width=500)

    report.save()
