import os
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import tensorflow as tf
import tflearn
import sys
import math
import random

from rllab.misc.instrument import run_experiment_lite
from rllab.misc.instrument import VariantGenerator
from sandbox.carlos_snn.autoclone import autoclone
from rllab import config

from sandbox.carlos_snn.runs.goal_rl.point_cl_gan_algo import run_task

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
        n_parallel = 4
    else:
        mode = 'local'
        n_parallel = 0

    exp_prefix = 'goalGAN-point11-singleLabel-test'

    vg = VariantGenerator()
    # # GeneratorEnv params
    vg.add('goal_size', [3])  # this is the ultimate goal we care about: getting the pendulum upright
    vg.add('reward_dist_threshold', lambda goal_size: [math.sqrt(goal_size) / math.sqrt(2) * 0.3])
    # vg.add('reward_dist_threshold', [0.5, 1])
    vg.add('goal_range', [5])  # this will be used also as bound of the state_space
    vg.add('state_bounds', lambda reward_dist_threshold, goal_range, goal_size:
    [(1, goal_range) + (reward_dist_threshold,) * (goal_size - 2) + (goal_range, ) * goal_size])
    vg.add('distance_metric', ['L2'])
    vg.add('dist_goal_weight', [0])
    vg.add('terminal_bonus', [300])
    vg.add('terminal_eps', lambda reward_dist_threshold: [
        reward_dist_threshold])  # if hte terminal bonus is 0 it doesn't kill it! Just count how many reached center
    #############################################
    vg.add('min_reward', lambda terminal_bonus: [terminal_bonus * 0.1])  # now running it with only the terminal reward of 1!
    vg.add('max_reward', lambda terminal_bonus: [terminal_bonus * 0.9])
    vg.add('improvement_threshold', lambda terminal_bonus: [terminal_bonus * 0.1])  # is this based on the reward, now discounted success rate --> push for fast
    vg.add('smart_init', [True])
    vg.add('replay_buffer', [False])
    vg.add('coll_eps', lambda reward_dist_threshold: [0])
    # old hyperparams
    vg.add('num_new_goals', [200])
    vg.add('num_old_goals', [100])
    vg.add('outer_iters', [400])
    vg.add('inner_iters', [5])
    vg.add('horizon', [200])
    vg.add('pg_batch_size', [20000])
    # policy initialization
    vg.add('output_gain', [1])  # check here if it goes wrong! both were 0.1
    vg.add('policy_init_std', [1])
    vg.add('learn_std', [True])
    vg.add('adaptive_std', [False])
    vg.add('seed', range(100, 200, 20))
    # gan_configs
    vg.add('num_labels', [1])
    vg.add('gan_generator_layers', [[256, 256]])
    vg.add('gan_discriminator_layers', [[128, 128]])
    vg.add('gan_noise_size', [4])
    vg.add('goal_noise_level', [0.5])  # ???
    vg.add('gan_outer_iters', [5])
    vg.add('gan_discriminator_iters', [200])
    vg.add('gan_generator_iters', [10])
    vg.add('GAN_batch_size', [128])  # proble with repeated name!!
    vg.add('GAN_print_iteration', [100])
    vg.add('GAN_generator_activation', ['relu'])
    vg.add('GAN_discriminator_activation', ['relu'])
    vg.add('GAN_generator_optimizer', [tf.train.AdamOptimizer])
    vg.add('GAN_generator_optimizer_stepSize', [0.001])
    vg.add('GAN_discriminator_optimizer', [tf.train.AdamOptimizer])
    vg.add('GAN_discriminator_optimizer_stepSize', [0.001])
    vg.add('GAN_generator_weight_initializer', [tflearn.initializations.truncated_normal])
    vg.add('GAN_generator_weight_initializer_stddev', [0.05])
    vg.add('GAN_discriminator_weight_initializer', [tflearn.initializations.truncated_normal])
    vg.add('GAN_discriminator_weight_initializer_stddev', [0.02])
    vg.add('GAN_discriminator_batch_noise_stddev', [1e-2])

    print('Running {} inst. on type {}, with price {}, parallel {} on the subnets: '.format(vg.size, config.AWS_INSTANCE_TYPE,
                                                                                   config.AWS_SPOT_PRICE, n_parallel),
          *subnets)
    for vv in vg.variants():
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
            )
