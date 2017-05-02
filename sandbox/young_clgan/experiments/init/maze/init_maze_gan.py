import argparse
import os
import datetime
from multiprocessing import cpu_count
import os.path as osp
import random
import sys
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tflearn

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Symbols that need to be stubbed
from rllab import config
from rllab.misc import logger
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import run_experiment_lite
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import VariantGenerator
from sandbox.carlos_snn.autoclone import autoclone

from sandbox.young_clgan.utils import format_experiment_prefix

from sandbox.young_clgan.logging import *
from sandbox.young_clgan.logging import HTMLReport, format_dict
from sandbox.young_clgan.logging.visualization import save_image, plot_labeled_samples
from sandbox.young_clgan.logging.inner_logger import InnerExperimentLogger

from sandbox.young_clgan.envs.maze.point_maze_env import PointMazeEnv
from sandbox.young_clgan.envs.init_sampler.base import InitIdxEnv, generate_initial_inits
from sandbox.young_clgan.state.generator import StateGAN
from sandbox.young_clgan.state.evaluator import label_states, convert_label
from sandbox.young_clgan.state.utils import StateCollection
from sandbox.young_clgan.state.selectors import UniformStateSelector, UniformListStateSelector, FixedStateSelector
from sandbox.young_clgan.envs.base import FixedStateGenerator  # kept for the point-mass env...

from sandbox.young_clgan.envs.maze.maze_evaluate import test_and_plot_policy  # this used for both init and goal

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])

    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=5)
    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    tf_session = tf.Session()

    inner_env = normalize(PointMazeEnv(
        goal_generator=FixedStateGenerator(v['goal']),
        reward_dist_threshold=v['reward_dist_threshold'] * 0.1,  # never stop from inner_env!
        append_goal=False,
    ))

    goal_selector = FixedStateSelector(state=v['goal'])
    init_selector = UniformStateSelector(state_size=np.size(v['goal']), bounds=v['init_range'],
                                         center=v['init_center'])

    env = InitIdxEnv(idx=range(2), env=inner_env, goal_selector=goal_selector, init_selector=init_selector,
                     goal_reward=v['goal_reward'], goal_weight=v['goal_weight'], terminal_bonus=v['terminal_bonus'],
                     inner_weight=v['inner_weight'], terminal_eps=v['terminal_eps'], persistence=v['persistence'])

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        learn_std=v['learn_std'],
        output_gain=v['output_gain'],
        init_std=v['policy_init_std'],
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    report.save()
    report.new_row()

    outer_itr = 0
    logger.log('Generating the Initial Heatmap...')
    avg_rewards, avg_success, heatmap = test_and_plot_policy(policy, env, as_goals=False, visualize=False,
                                                             sampling_res=v['sampling_res'], n_traj=v['n_traj'])
    reward_img = save_image()

    mean_rewards = np.mean(avg_rewards)
    mean_success = np.mean(avg_success)

    with logger.tabular_prefix('Outer_'):
        logger.record_tabular('iter', outer_itr)
        logger.record_tabular('MeanRewards', mean_rewards)
        logger.record_tabular('Success', mean_success)
    # logger.dump_tabular(with_prefix=False)

    report.add_image(
        reward_img,
        'policy performance\n itr: {} \nmean_rewards: {} \nsuccess: {}'.format(
            outer_itr, mean_rewards, mean_success
        )
    )

    # GAN
    logger.log("Instantiating the GAN...")
    gan_configs = {key[4:]: value for key, value in v.items() if 'GAN_' in key}
    for key, value in gan_configs.items():
        if value is tf.train.AdamOptimizer:
            gan_configs[key] = tf.train.AdamOptimizer(gan_configs[key + '_stepSize'])
        if value is tflearn.initializations.truncated_normal:
            gan_configs[key] = tflearn.initializations.truncated_normal(stddev=gan_configs[key + '_stddev'])

    final_gen_loss = 11
    k = -1
    while final_gen_loss > 10:
        k += 1
        gan = StateGAN(
            state_size=np.size(v['goal']),
            evaluater_size=v['num_labels'],
            state_range=v['init_range'],
            state_center=v['init_center'],
            state_noise_level=v['init_noise_level'],
            generator_layers=v['gan_generator_layers'],
            discriminator_layers=v['gan_discriminator_layers'],
            noise_size=v['gan_noise_size'],
            tf_session=tf_session,
            configs=gan_configs,
        )
        logger.log("pretraining the GAN...")
        if v['smart_init']:
            dis_loss, gen_loss = gan.pretrain(
                generate_initial_inits(env, policy, max_path_length=v['max_path_length']),
                outer_iters=30, generator_iters=10 + k, discriminator_iters=200 - k * 10,
            )
            final_gen_loss = gen_loss[-1]
            logger.log("error at the end of {}th trial: {}gen, {}disc".format(k, gen_loss[-1], dis_loss[-1]))
        else:
            gan.pretrain_uniform()
            final_gen_loss = 0

    # log first samples form the GAN
    initial_inits, _ = gan.sample_states_with_noise(v['num_new_states'])
    logger.log("Labeling the goals")
    labels = label_states(
        initial_inits, env, policy, v['max_path_length'],
        min_reward=v['min_reward'],
        max_reward=v['max_reward'],
        as_goals=False,
        old_rewards=None,
        n_traj=v['n_traj'])

    logger.log("Converting the labels")
    init_classes, text_labels = convert_label(labels)

    logger.log("Plotting the labeled samples")
    total_goals = labels.shape[0]
    init_class_frac = OrderedDict()  # this needs to be an ordered dict!! (for the log tabular)
    for k in text_labels.keys():
        frac = np.sum(init_classes == k) / total_goals
        logger.record_tabular('TrainInit_frac_' + text_labels[k], frac)
        logger.record_tabular('GenInit_frac_' + text_labels[k], frac)
        init_class_frac[text_labels[k]] = frac

    img = plot_labeled_samples(
        samples=initial_inits, sample_classes=init_classes, text_labels=text_labels,
        limit=v['init_range'] + 0.5, center=v['init_center']
        # '{}/sampled_goals_{}.png'.format(log_dir, outer_iter),  # if i don't give the file it doesn't save
    )
    summary_string = ''
    for key, value in init_class_frac.items():
        summary_string += key + ' frac: ' + str(value) + '\n'

    report.add_image(img, 'itr: {}\nLabels of generated goals:\n{}'.format(outer_itr, summary_string),
                     width=500)

    report.save()
    report.new_row()
    logger.dump_tabular(with_prefix=False)

    all_inits = StateCollection(distance_threshold=v['coll_eps'])

    inner_experiment_logger = InnerExperimentLogger(log_dir, 'inner', snapshot_mode='last', hold_outter_log=True)

    for outer_itr in range(1, v['outer_itr']):
        logger.log("Outer itr # %i" % outer_itr)
        # Sample GAN
        logger.log("Sampling inits from the GAN")
        raw_inits, _ = gan.sample_states_with_noise(v['num_new_states'])

        if v['replay_buffer'] and outer_itr > 0 and all_inits.size > 0:
            old_inits = all_inits.sample(v['num_old_states'], replay_noise=v['replay_noise'])
            inits = np.vstack([raw_inits, old_inits])
        else:
            inits = raw_inits

        with inner_experiment_logger:
            logger.log("Updating the environment init generator")
            env.update_init_selector(UniformListStateSelector(inits.tolist()))

            logger.log('Training the algorithm')
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=v['batch_size'],
                max_path_length=v['max_path_length'],
                n_itr=v['inner_itr'],
                gae_lambda=v['gae_lambda'],
                discount=v['discount'],
                step_size=0.01,
                plot=False,
            )

            algo.train()

        logger.log("Plot states trained on")
        report.new_row()
        states_by_classes = OrderedDict()  # initialize global count
        for key in range(5):  # TODO: this is now hard-coded to 5 labels, should get it by itself.
            states_by_classes[key] = 0
        for i, iter_samples in enumerate(env.inits_trained):
            states = np.array(list(iter_samples.keys()))
            mean_rewards = np.array([np.mean(rewards) for rewards in iter_samples.values()]).reshape(-1, 1)
            labels = np.hstack(
                [mean_rewards > v['min_reward'], mean_rewards < v['max_reward']]
            ).astype(np.float32)
            init_classes, text_labels = convert_label(labels)
            # compute stat on these labels
            total_inits = labels.shape[0]
            init_class_frac = OrderedDict()  # this needs to be an ordered dict!! (for the log tabular)
            for k in text_labels.keys():
                num_class = np.sum(init_classes == k)
                states_by_classes[k] += num_class  # keep track of total of each sampled for logging
                frac = num_class / total_inits
                init_class_frac[text_labels[k]] = frac
            img = plot_labeled_samples(
                samples=states, sample_classes=init_classes, text_labels=text_labels,
                limit=v['init_range'] + 0.5, center=v['init_center']
                # '{}/sampled_goals_{}.png'.format(log_dir, outer_iter),  # if i don't give the file it doesn't save
            )
            summary_string = ''
            for key, value in init_class_frac.items():
                summary_string += key + ' frac: ' + str(value) + '\n'
            report.add_image(img, 'itr: {}, inner_itr: {}\nLabels of pre-trained inits:\n{}'.format(
               outer_itr, i, summary_string),
                             width=500)
        report.new_row()
        # log stats accross all itr
        total_inits = np.sum([v for v in states_by_classes.values()])
        for key, value in states_by_classes.items():
            logger.record_tabular('TrainInit_frac_' + text_labels[key], value/total_inits)
        env.inits_trained = []

        logger.log('Generating the Heatmap...')
        avg_rewards, avg_success, heatmap = test_and_plot_policy(policy, env, as_goals=False, visualize=False,
                                                                 sampling_res=v['sampling_res'], n_traj=v['n_traj'])
        reward_img = save_image()

        mean_rewards = np.mean(avg_rewards)
        mean_success = np.mean(avg_success)

        with logger.tabular_prefix('Outer_'):
            logger.record_tabular('iter', outer_itr)
            logger.record_tabular('MeanRewards', mean_rewards)
            logger.record_tabular('Success', mean_success)

        report.add_image(
            reward_img,
            'policy performance\n itr: {} \nmean_rewards: {} \nsuccess: {}'.format(
                outer_itr, mean_rewards, mean_success,
            )
        )

        logger.log("Labeling the goals")
        labels = label_states(
            inits, env, policy, v['max_path_length'],
            min_reward=v['min_reward'],
            max_reward=v['max_reward'],
            as_goals=False,
            old_rewards=None,
            n_traj=v['n_traj'])
        init_classes, text_labels = convert_label(labels)

        logger.log("Plotting the labeled samples")
        total_inits = labels.shape[0]
        init_class_frac = OrderedDict()  # this needs to be an ordered dict!! (for the log tabular)
        for k in text_labels.keys():
            frac = np.sum(init_classes == k) / total_inits
            logger.record_tabular('GenInit_frac_' + text_labels[k], frac)
            init_class_frac[text_labels[k]] = frac

        img = plot_labeled_samples(
            samples=inits, sample_classes=init_classes, text_labels=text_labels,
            limit=v['init_range'] + 0.5, center=v['init_center']
            # '{}/sampled_goals_{}.png'.format(log_dir, outer_iter),  # if i don't give the file it doesn't save
        )
        summary_string = ''
        for key, value in init_class_frac.items():
            summary_string += key + ' frac: ' + str(value) + '\n'
        report.add_image(img, 'itr: {}\nLabels of generated inits:\n{}'.format(outer_itr, summary_string),
                         width=500)

        if v['num_labels'] == 1:
            labels = np.logical_and(labels[:, 0], labels[:, 1]).astype(int).reshape((-1, 1))

        logger.log("Training the GAN")
        gan.train(
            inits, labels,
            v['gan_outer_iters'],
        )

        logger.dump_tabular(with_prefix=False)
        report.save()
        report.new_row()

        # append new inits to list of all inits (replay buffer): Not the low reward ones!!
        filtered_raw_inits = [init for init, label in zip(inits, labels) if label[0] == 2]
        all_inits.append(filtered_raw_inits)


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
    args = parser.parse_args()

    if args.clone:
        autoclone.autoclone(__file__, args)

    # setup ec2
    subnets = [
        'us-east-2b', 'us-east-1a', 'us-east-1d', 'us-east-1b', 'us-east-1e', 'ap-south-1b', 'ap-south-1a', 'us-west-1a'
    ]
    ec2_instance = args.type if args.type else 'c4.2xlarge'

    # configure instance
    info = config.INSTANCE_TYPE_INFO[ec2_instance]
    config.AWS_INSTANCE_TYPE = ec2_instance
    config.AWS_SPOT_PRICE = str(info["price"])
    n_parallel = int(info["vCPU"])  # make the default 4 if not using ec2
    if args.ec2:
        mode = 'ec2'
    elif args.local_docker:
        mode = 'local_docker'
        n_parallel = cpu_count()
    else:
        mode = 'local'
        n_parallel = cpu_count()

    exp_prefix = format_experiment_prefix('init-maze-gan')
    vg = VariantGenerator()
    vg.add('test', [False])
    vg.add('n_traj', lambda test: [3])
    vg.add('persistence', lambda n_traj: [1])
    vg.add('sampling_res', lambda test: [1] if test else [2])
    # algorithm params
    vg.add('seed', range(21, 80, 20))
    vg.add('n_itr', [500])
    vg.add('inner_itr', lambda test: [5] if test else [5, 3])
    vg.add('outer_itr', lambda n_itr, inner_itr: [int(n_itr / inner_itr)])
    vg.add('batch_size', [20000])
    vg.add('max_path_length', [400])
    # environemnt params
    vg.add('init_center', [(2, 2)])
    vg.add('init_range', [4])
    vg.add('goal', [(0, 4), ])
    vg.add('goal_reward', ['NegativeDistance'])
    vg.add('goal_weight', [0])  # this makes the task spars
    vg.add("inner_weight", [0])
    vg.add('terminal_bonus', [1])
    vg.add('distance_metric', ['L2'])
    vg.add('reward_dist_threshold', [0.3])
    vg.add('terminal_eps', lambda reward_dist_threshold: [reward_dist_threshold])
    vg.add('indicator_reward', [True])
    vg.add('max_reward', [0.9])
    vg.add('min_reward', [0.1])
    # policy hypers
    vg.add('learn_std', [True])
    vg.add('policy_init_std', [1])
    vg.add('output_gain', [1])
    # algo params
    vg.add('smart_init', [False])
    vg.add('replay_buffer', [True])
    vg.add('coll_eps', [0.3])  # lambda reward_dist_threshold: [reward_dist_threshold, 0])
    vg.add('replay_noise', [0, 0.1])

    vg.add('discount', [0.995])
    vg.add('gae_lambda', [1])
    vg.add('num_labels', [1])  # 1 for single label, 2 for high/low and 3 for learnability

    vg.add('init_noise_level', [0.8])
    vg.add('gan_outer_iters', [50])
    # vg.add('gan_discriminator_iters', [2])
    # vg.add('gan_generator_iters', [1])
    vg.add('gan_noise_size', [4])
    vg.add('num_new_states', [200])
    vg.add('num_old_states', [100])
    vg.add('gan_generator_layers', [[256, 256]])
    vg.add('gan_discriminator_layers', [[128, 128]])
    # gan_configs
    vg.add('GAN_batch_size', [128])  # proble with repeated name!!
    # vg.add('GAN_generator_activation', ['relu'])
    # vg.add('GAN_discriminator_activation', ['relu'])
    # vg.add('GAN_generator_optimizer', [tf.train.AdamOptimizer])
    # vg.add('GAN_generator_optimizer_stepSize', [0.001])
    # vg.add('GAN_discriminator_optimizer', [tf.train.AdamOptimizer])
    # vg.add('GAN_discriminator_optimizer_stepSize', [0.001])
    # vg.add('GAN_generator_weight_initializer', [tflearn.initializations.truncated_normal])
    # vg.add('GAN_generator_weight_initializer_stddev', [0.05])
    # vg.add('GAN_discriminator_weight_initializer', [tflearn.initializations.truncated_normal])
    # vg.add('GAN_discriminator_weight_initializer_stddev', [0.02])
    # vg.add('GAN_discriminator_batch_noise_stddev', [1e-2])

    # print('Running {} inst. on type {}, with price {}, parallel {} on the subnets: '.format(vg.size, config.AWS_INSTANCE_TYPE,
    #                                                                                         config.AWS_SPOT_PRICE, n_parallel),
    #       *subnets)
          
    print(
        'Running {} inst. on type {}, with price {}, parallel {}'.format(
            vg.size,
            config.AWS_INSTANCE_TYPE,
            config.AWS_SPOT_PRICE,
            n_parallel
        )
    )

    for vv in vg.variants(randomized=False):

        if mode in ['ec2', 'local_docker']:
            # # choose subnet
            # subnet = random.choice(subnets)
            # config.AWS_REGION_NAME = subnet[:-1]
            # config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[
            #     config.AWS_REGION_NAME]
            # config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[
            #     config.AWS_REGION_NAME]
            # config.AWS_SECURITY_GROUP_IDS = \
            #     config.ALL_REGION_AWS_SECURITY_GROUP_IDS[
            #         config.AWS_REGION_NAME]
            # config.AWS_NETWORK_INTERFACES = [
            #     dict(
            #         SubnetId=config.ALL_SUBNET_INFO[subnet]["SubnetID"],
            #         Groups=config.AWS_SECURITY_GROUP_IDS,
            #         DeviceIndex=0,
            #         AssociatePublicIpAddress=True,
            #     )
            # ]

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
                sync_s3_pkl=True,
                sync_s3_html=True,
                # for sync the pkl file also during the training
                sync_s3_png=True,
                # # use this ONLY with ec2 or local_docker!!!
                pre_commands=[
                    'export MPLBACKEND=Agg',
                    'pip install --upgrade pip',
                    'pip install --upgrade -I tensorflow',
                    'pip install git+https://github.com/tflearn/tflearn.git',
                    'pip install dominate',
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
