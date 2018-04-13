from curriculum.utils import set_env_no_gpu, format_experiment_prefix

set_env_no_gpu()

import argparse
import tflearn
import sys
import math
import os
import os.path as osp
import random
from multiprocessing import cpu_count

import tensorflow as tf

from rllab.misc.instrument import run_experiment_lite
from rllab.misc.instrument import VariantGenerator
from rllab import config

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from curriculum.envs.ndim_point.point_env import PointEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from curriculum.envs.goal_env import GoalExplorationEnv, evaluate_goal_env, generate_initial_goals
from curriculum.envs.base import FixedStateGenerator, UniformStateGenerator, UniformListStateGenerator

from curriculum.state.evaluator import *
from curriculum.state.generator import StateGAN
# from curriculum.lib.goal.utils import *
from curriculum.logging.html_report import format_dict, HTMLReport
from curriculum.logging.visualization import *
from curriculum.logging.logger import ExperimentLogger
from curriculum.state.utils import StateCollection
from curriculum.experiments.goals.point_nd.utils import plot_policy_performance, plot_generator_samples

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])

    # goal generators
    logger.log("Initializing the goal generators and the inner env...")
    inner_env = normalize(PointEnv(dim=v['goal_size'], state_bounds=v['state_bounds']))
    # inner_env = normalize(PendulumEnv())

    center = np.zeros(v['goal_size'])
    uniform_goal_generator = UniformStateGenerator(state_size=v['goal_size'], bounds=v['goal_range'],
                                                   center=center)
    feasible_goal_ub = np.array(v['state_bounds'])[:v['goal_size']]
    # print("the feasible_goal_ub is: ", feasible_goal_ub)
    uniform_feasible_goal_generator = UniformStateGenerator(
        state_size=v['goal_size'], bounds=[-1 * feasible_goal_ub, feasible_goal_ub]
    )

    env = GoalExplorationEnv(
        env=inner_env, goal_generator=uniform_goal_generator,
        obs2goal_transform=lambda x: x[:int(len(x) / 2)],
        terminal_eps=v['terminal_eps'],
        only_feasible=v['only_feasible'],
        distance_metric=v['distance_metric'],
        terminate_env=True, goal_weight=v['goal_weight'],
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
    n_traj = 3

    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()
    inner_log_dir = osp.join(log_dir, 'inner_iters')
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

    gan = StateGAN(
        state_size=v['goal_size'],
        evaluater_size=v['num_labels'],
        state_range=v['goal_range'],
        state_noise_level=v['goal_noise_level'],
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
                initial_goals, outer_iters=30
                # initial_goals, outer_iters=30, generator_iters=10, discriminator_iters=200,
            )
            final_gen_loss = gen_loss
            logger.log("Loss at the end of {}th trial: {}gen, {}disc".format(k, gen_loss, dis_loss))
        else:
            gan.pretrain_uniform()
            final_gen_loss = 0
        logger.log("Plotting GAN samples")
        img = plot_gan_samples(gan, v['goal_range'], '{}/start.png'.format(log_dir))
        report.add_image(img, 'GAN pretrained %i: %i gen_itr, %i disc_itr' % (k, 10 + k, 200 - k * 10))
        # report.add_image(img, 'GAN pretrained %i: %i gen_itr, %i disc_itr' % (k, 10, 200))
        report.save()
        report.new_row()

    all_goals = StateCollection(v['coll_eps'])

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
            rewards_before = evaluate_states(goals, env, policy, v['horizon'], n_traj=n_traj)

        logger.log("Perform TRPO with UniformListStateGenerator...")
        with ExperimentLogger(inner_log_dir, '_last', snapshot_mode='last', hold_outter_log=True):
            # set goal generator to uniformly sample from selected all_goals
            env.update_goal_generator(
                UniformListStateGenerator(
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
        
        report.add_image(
            plot_generator_samples(gan, env), 'policy_rewards_{}'.format(outer_iter)
        )
        
        report.add_image(
            plot_policy_performance(policy, env, v['horizon']),
            'gan_samples_{}'.format(outer_iter)
        )

        # this re-evaluate the final policy in the collection of goals
        logger.log("Generating labels by re-evaluating policy on List of goals...")
        labels = label_states(
            goals, env, policy, v['horizon'],
            min_reward=v['min_reward'],
            max_reward=v['max_reward'],
            old_rewards=rewards_before,
            # improvement_threshold=0,
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
            bounds=env.feasible_goal_space.bounds,
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
        )

        logger.log("Evaluating performance on Unif and Fix Goal Gen...")
        with logger.tabular_prefix('UnifFeasGoalGen_'):
            env.update_goal_generator(uniform_feasible_goal_generator)
            evaluate_goal_env(env, policy=policy, horizon=v['horizon'], n_goals=50, fig_prefix='UnifFeasGoalGen_',
                              report=report, n_traj=n_traj)

        logger.dump_tabular(with_prefix=False)

        report.save()
        report.new_row()

        # append new goals to list of all goals (replay buffer): Not the low reward ones!!
        filtered_raw_goals = [goal for goal, label in zip(goals, labels) if label[0] == 1]
        all_goals.append(filtered_raw_goals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ec2', '-e', action='store_true', default=False, help="add flag to run in ec2")
    parser.add_argument('--clone', '-c', action='store_true', default=False,
                        help="add flag to copy file and checkout current")
    parser.add_argument('--local_docker', '-d', action='store_true', default=False,
                        help="add flag to run in local dock")
    parser.add_argument('--type', '-t', type=str, default='', help='set instance type')
    parser.add_argument('--price', '-p', type=str, default='', help='set betting price')
    parser.add_argument('--subnet', '-sn', type=str, default='', help='set subnet like us-west-1a')
    parser.add_argument('--name', '-n', type=str, default='', help='set exp prefix name and new file name')
    parser.add_argument('--debug', action='store_true', default=False, help="run code without multiprocessing")
    parser.add_argument(
        '--prefix', type=str, default=None,
        help='set the additional name for experiment prefix'
    )
    args = parser.parse_args()



    # setup ec2
    subnets = [
        'us-east-2c', 'us-east-2b', 'us-east-2a', 'ap-southeast-2b', 'ap-southeast-1b', 'ap-southeast-2c', 'us-west-2c',
        'ap-southeast-1a', 'eu-west-1a', 'us-west-1a', 'us-east-1b', 'us-west-1b', 'eu-west-1b',
        'ap-northeast-1a'
    ]
    ec2_instance = args.type if args.type else 'm4.4xlarge'
    # configure instance
    info = config.INSTANCE_TYPE_INFO[ec2_instance]
    config.AWS_INSTANCE_TYPE = ec2_instance
    config.AWS_SPOT_PRICE = str(info["price"])
    n_parallel = int(info["vCPU"] / 2)  # make the default 4 if not using ec2
    if args.ec2:
        mode = 'ec2'
    elif args.local_docker:
        mode = 'local_docker'
        n_parallel = cpu_count() if not args.debug else 1
    else:
        mode = 'local'
        n_parallel = cpu_count() if not args.debug else 1

    default_prefix = 'point-nd-goal-gan'
    if args.prefix is None:
        exp_prefix = format_experiment_prefix(default_prefix)
    elif args.prefix == '':
        exp_prefix = default_prefix
    else:
        exp_prefix = '{}_{}'.format(default_prefix, args.prefix)

    vg = VariantGenerator()
    # # GeneratorEnv params
    vg.add('goal_size', [2, 3, 4, 5, 6])
    vg.add('terminal_eps', lambda goal_size: [math.sqrt(goal_size) / math.sqrt(2) * 0.3])
    vg.add('only_feasible', [True])
    vg.add('goal_range', [5])  # this will be used also as bound of the state_space
    vg.add('state_bounds', lambda goal_range, goal_size, terminal_eps:
    [(1, goal_range) + (0.3,) * (goal_size - 2) + (goal_range,) * goal_size])
    vg.add('distance_metric', ['L2'])
    vg.add('goal_weight', [1])
    #############################################
    # goal-algo params
    vg.add('min_reward', lambda goal_weight: [goal_weight * 0.1])  # now running it with only the terminal reward of 1!
    vg.add('max_reward', lambda goal_weight: [goal_weight * 0.9])
    vg.add('smart_init', [True])
    # replay buffer
    vg.add('replay_buffer', [True])
    vg.add('coll_eps', [0.3])  # lambda terminal_eps: [terminal_eps])
    vg.add('num_new_goals', [200])
    vg.add('num_old_goals', [100])
    # sampling params
    vg.add('horizon', [200])
    vg.add('outer_iters', [200])
    vg.add('inner_iters', [5])
    vg.add('pg_batch_size', [20000])
    # policy params
    vg.add('output_gain', [1])  # check here if it goes wrong! both were 0.1
    vg.add('policy_init_std', [1])
    vg.add('learn_std', [True])
    vg.add('adaptive_std', [False])
    # gan_configs
    vg.add('num_labels', [1])
    vg.add('gan_generator_layers', [[256, 256]])
    vg.add('gan_discriminator_layers', [[128, 128]])
    vg.add('gan_noise_size', [4])
    vg.add('goal_noise_level', [0.5])
    vg.add('gan_outer_iters', [100])

    vg.add('seed', range(100, 200, 20))

    print('\n****\nRunning {} inst. on type {}, with price {}, parallel {} on the subnets: '.format(
        vg.size, config.AWS_INSTANCE_TYPE, config.AWS_SPOT_PRICE, n_parallel),
        *subnets)
    for vv in vg.variants():
        if mode in ['ec2', 'local_docker']:

            run_experiment_lite(
                # use_cloudpickle=False,
                stub_method_call=run_task,
                variant=vv,
                mode=mode,
                # Number of parallel workers for sampling
                n_parallel=n_parallel,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                seed=vv['seed'],
                # plot=True,
                exp_prefix=exp_prefix,
                # exp_name=exp_name,
                # for sync the pkl file also during the training
                sync_s3_pkl=True,
                # sync_s3_png=True,
                sync_s3_html=True,
                # # use this ONLY with ec2 or local_docker!!!
                pre_commands=[
                    'export MPLBACKEND=Agg',
                    'pip install --upgrade pip',
                    'pip install --upgrade -I tensorflow',
                    'pip install git+https://github.com/tflearn/tflearn.git',
                    'pip install dominate',
                    'pip install multiprocessing_on_dill',
                    'pip install scikit-image',
                    'conda install numpy -n rllab3 -y',
                ],
            )
            if mode == 'local_docker':
                sys.exit()
        else:
            run_experiment_lite(
                # use_cloudpickle=False,
                stub_method_call=run_task,
                variant=vv,
                mode='local',
                n_parallel=n_parallel,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                seed=vv['seed'],
                exp_prefix=exp_prefix,
                # exp_name=exp_name,
                print_command=False,
            )
            if args.debug:
                sys.exit()
