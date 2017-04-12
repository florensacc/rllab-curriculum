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
    update_env_goal_generator, generate_initial_goals

from sandbox.young_clgan.goal.evaluator import *
from sandbox.young_clgan.goal.generator import StateGAN
# from sandbox.young_clgan.lib.goal.utils import *
from sandbox.young_clgan.logging.html_report import format_dict, HTMLReport
from sandbox.young_clgan.logging.visualization import *
from sandbox.young_clgan.logging.logger import ExperimentLogger
from sandbox.young_clgan.goal.utils import GoalCollection

from sandbox.young_clgan.utils import initialize_parallel_sampler

initialize_parallel_sampler()

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])

    # goal generators
    logger.log("Initializing the goal generators and the inner env...")
    inner_env = normalize(PointEnv(dim=v['goal_size'], state_bounds=v['state_bounds']))
    # inner_env = normalize(PendulumEnv())

    center = np.zeros(v['goal_size'])
    fixed_goal_generator = FixedGoalGenerator(goal=center)
    # uniform_goal_generator = UniformGoalGenerator(goal_size=v['goal_size'], bounds=v['goal_range'],
    #                                               center=center)
    feasible_goal_ub = np.array(v['state_bounds'])[:v['goal_size']]
    # print("the feasible_goal_ub is: ", feasible_goal_ub)
    uniform_feasible_goal_generator = UniformGoalGenerator(goal_size=v['goal_size'], bounds=[-1 * feasible_goal_ub,
                                                                                             feasible_goal_ub])

    env = GoalIdxExplorationEnv(env=inner_env, goal_generator=fixed_goal_generator,
                                idx=np.arange(v['goal_size']),
                                reward_dist_threshold=v['reward_dist_threshold'],
                                distance_metric=v['distance_metric'],
                                dist_goal_weight=v['dist_goal_weight'], max_reward=v['max_reward'],
                                terminal_eps=v['terminal_eps'], terminal_bonus=v['terminal_bonus'],
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
    n_traj = 3 if v['terminal_bonus'] > 0 else 1

    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=3)
    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    # img = plot_policy_reward(
    #     policy, env, v['goal_range'],
    #     horizon=v['horizon'], grid_size=5,
    #     fname='{}/policy_reward_init.png'.format(log_dir),
    # )
    # report.add_image(img, 'policy performance initialization\n')

    # GAN
    logger.log("Instantiating the GAN...")
    tf_session = tf.Session()
    gan_configs = {key[4:]: value for key, value in v.items() if 'GAN_' in key}
    for key, value in gan_configs.items():
        if value is tf.train.AdamOptimizer or isinstance(value, tf.train.AdamOptimizer):
            gan_configs[key] = tf.train.AdamOptimizer(gan_configs[key + '_stepSize'])
        if 'initializer' in key and 'stddev' not in key:  # value is tflearn.initializations.truncated_normal: or isinstance(value, tf.initializations.truncated_normal):
            gan_configs[key] = tflearn.initializations.truncated_normal(stddev=gan_configs[key + '_stddev'])
    gan = StateGAN(
        goal_size=v['goal_size'],
        evaluater_size=v['num_labels'],
        goal_range=v['goal_range'],
        goal_noise_level=v['goal_noise_level'],
        generator_layers=v['gan_generator_layers'],
        discriminator_layers=v['gan_discriminator_layers'],
        noise_size=v['gan_noise_size'],
        tf_session=tf_session,
        configs=gan_configs,
    )

    final_gen_loss = 11
    k = -1
    while final_gen_loss > 10:
        k += 1
        gan.gan.initialize()
        img = plot_gan_samples(gan, v['goal_range'], '{}/start.png'.format(log_dir))
        report.add_image(img, 'GAN re-initialized %i' % k)
        logger.log("pretraining the GAN...")
        if v['smart_init']:
            initial_goals = generate_initial_goals(env, policy, v['goal_range'], horizon=v['horizon'])
            if np.size(initial_goals[0]) == 2:
                plt.figure()
                plt.scatter(initial_goals[:, 0], initial_goals[:, 1], marker='x')
                plt.xlim(-v['goal_range'], v['goal_range'])
                plt.ylim(-v['goal_range'], v['goal_range'])
                img = save_image()
                report.add_image(img, 'goals sampled to pretrain GAN: {}'.format(np.shape(initial_goals)))
            dis_loss, gen_loss = gan.pretrain(
                initial_goals, outer_iters=30, generator_iters=10 + k, discriminator_iters=200 - k * 10,
                # initial_goals, outer_iters=30, generator_iters=10, discriminator_iters=200,
            )
            final_gen_loss = gen_loss[-1]
            logger.log("error at the end of {}th trial: {}gen, {}disc".format(k, gen_loss[-1], dis_loss[-1]))
        else:
            gan.pretrain_uniform()
            final_gen_loss = 0
        logger.log("Plotting GAN samples")
        img = plot_gan_samples(gan, v['goal_range'], '{}/start.png'.format(log_dir))
        report.add_image(img, 'GAN pretrained %i: %i gen_itr, %i disc_itr' % (k, 10 + k, 200 - k * 10))
        # report.add_image(img, 'GAN pretrained %i: %i gen_itr, %i disc_itr' % (k, 10, 200))
        report.save()
        report.new_row()

    all_goals = GoalCollection(v['coll_eps'])

    logger.log("Starting the outer iterations")
    for outer_iter in range(v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)
        # Train GAN
        logger.log("Sampling goals...")
        raw_goals, _ = gan.sample_states_with_noise(v['num_new_goals'])

        if v['replay_buffer'] and outer_iter > 0 and all_goals.size > 0:
            # sampler uniformly 2000 old goals and add them to the training pool (50/50)
            old_goals = all_goals.sample(v['num_old_goals'])
            # print("old_goals: {}, raw_goals: {}".format(old_goals, raw_goals))
            goals = np.vstack([raw_goals, old_goals])
        else:
            # print("no goals in all_goals: sample fresh ones")
            goals = raw_goals

        logger.log("Evaluating goals before inner training...")

        rewards_before = None
        if v['num_labels'] == 3:
            rewards_before = evaluate_goals(goals, env, policy, v['horizon'], n_traj=n_traj,
                                            n_processes=multiprocessing.cpu_count())

        logger.log("Perform TRPO with UniformListGoalGenerator...")
        with ExperimentLogger(log_dir, outer_iter, snapshot_mode='last', hold_outter_log=True):
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
                discount=0.995,
                step_size=0.01,
                plot=False,
            )

            algo.train()

        # logger.log("Plot performance policy on full grid...")
        # img = plot_policy_reward(
        #     policy, env, v['goal_range'],
        #     horizon=v['horizon'],
        #     max_reward=v['max_reward'],
        #     grid_size=10,
        #     # fname='{}/policy_reward_{}.png'.format(log_config.plot_dir, outer_iter),
        # )
        # report.add_image(img, 'policy performance\n itr: {}'.format(outer_iter))
        # report.save()

        # this re-evaluate the final policy in the collection of goals
        logger.log("Generating labels by re-evaluating policy on List of goals...")
        labels = label_states(
            goals, env, policy, v['horizon'],
            min_reward=v['min_reward'],
            max_reward=v['max_reward'],
            old_rewards=rewards_before,
            improvement_threshold=v['improvement_threshold'],
            n_traj=n_traj,
        )
        goal_classes, text_labels = convert_label(labels)
        total_goals = labels.shape[0]
        goal_class_frac = OrderedDict()  # this needs to be an ordered dict!! (for the log tabular)
        for k in text_labels.keys():
            frac = np.sum(goal_classes == k) / total_goals
            logger.record_tabular('GenGoal_frac_' + text_labels[k], frac)
            goal_class_frac[text_labels[k]] = frac

        img = plot_labeled_samples(
            samples=goals, sample_classes=goal_classes, text_labels=text_labels, limit=v['goal_range'] + 1,
            # '{}/sampled_goals_{}.png'.format(log_dir, outer_iter),  # if i don't give the file it doesn't save
        )
        summary_string = ''
        for key, value in goal_class_frac.items():
            summary_string += key + ' frac: ' + str(value) + '\n'
        report.add_image(img, 'itr: {}\nLabels of generated goals:\n{}'.format(outer_iter, summary_string), width=500)

        # log feasibility of generated goals
        feasible = np.array([1 if env.feasible_goal_space.contains(goal) else 0 for goal in goals], dtype=int)
        feasibility_rate = np.mean(feasible)
        logger.record_tabular('GenGoalFeasibilityRate', feasibility_rate)
        img = plot_labeled_samples(
            samples=goals, sample_classes=feasible, text_labels={0: 'Infeasible', 1: "Feasible"},
            markers={0: 'v', 1: 'o'}, limit=v['goal_range'] + 1, bounds=env.feasible_goal_space.bounds,
            # '{}/sampled_goals_{}.png'.format(log_dir, outer_iter),  # if i don't give the file it doesn't save
        )
        report.add_image(img, 'feasibility of generated goals: {}\n itr: {}'.format(feasibility_rate, outer_iter),
                         width=500)

        ######  try single label for good goals
        if v['num_labels'] == 1:
            labels = np.logical_and(labels[:, 0], labels[:, 1]).astype(int).reshape((-1, 1))

        logger.log("Training GAN...")
        gan.train(
            goals, labels,
            v['gan_outer_iters'],
            v['gan_generator_iters'],
            v['gan_discriminator_iters']
        )

        logger.log("Evaluating performance on Unif and Fix Goal Gen...")
        with logger.tabular_prefix('UnifFeasGoalGen_'):
            update_env_goal_generator(env, goal_generator=uniform_feasible_goal_generator)
            evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=50, fig_prefix='UnifFeasGoalGen_',
                              report=report, n_traj=n_traj)
        # with logger.tabular_prefix('FixGoalGen_'):
        #     update_env_goal_generator(env, goal_generator=fixed_goal_generator)
        #     evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=5, fig_prefix='FixGoalGen',
        #                       report=report)

        logger.dump_tabular(with_prefix=False)

        report.save()
        report.new_row()

        # append new goals to list of all goals (replay buffer): Not the low reward ones!!
        filtered_raw_goals = [goal for goal, label in zip(goals, labels) if label[0] == 1]
        all_goals.append(filtered_raw_goals)

    with logger.tabular_prefix('FINALUnifFeasGoalGen_'):
        update_env_goal_generator(env, goal_generator=uniform_feasible_goal_generator)
        evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=5e3, fig_prefix='FINAL1UnifFeasGoalGen_',
                          report=report, n_traj=n_traj)
        evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=5e3, fig_prefix='FINAL2UnifFeasGoalGen_',
                          report=report, n_traj=n_traj)
    logger.dump_tabular(with_prefix=False)
