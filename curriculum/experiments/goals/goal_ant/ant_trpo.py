import os
import random

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import tflearn
import argparse
import sys
from multiprocessing import cpu_count
from rllab.misc.instrument import run_experiment_lite
from rllab.misc.instrument import VariantGenerator

from rllab import config

from curriculum.experiments.goals.goal_ant.ant_trpo_algo import run_task

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
    args = parser.parse_args()



    # setup ec2
    subnets = [
        'ap-southeast-2b', 'ap-southeast-2c',
        'ap-northeast-1c', 'eu-west-1c',
        'eu-central-1b', 'ap-southeast-1a', 'us-east-1a'
    ]
    ec2_instance = args.type if args.type else 'm4.4xlarge'
    # configure instan
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
        # n_parallel = multiprocessing.cpu_count()

    exp_prefix = 'trpo-ant-goal2'

    vg = VariantGenerator()
    vg.add('goal_size', [2])  # this is the ultimate goal we care about: getting the pendulum upright
    vg.add('terminal_eps', [0.5])
    # vg.add('only_feasible', [True])
    # vg.add('maze_id', [0])
    vg.add('goal_range', [5])
    vg.add('goal_center', lambda goal_range: [(0, 0)] if goal_range == 5 else [(0, 2)])
    # goal-algo params
    vg.add('min_reward', [0.1])
    vg.add('max_reward', [0.9])
    vg.add('distance_metric', ['L2'])
    vg.add('extend_dist_rew', [False, True])  # !!!!
    vg.add('persistence', [1])
    vg.add('n_traj', [3])  # only for labeling and plotting (for now, later it will have to be equal to persistence!)
    vg.add('with_replacement', [False])
    vg.add('use_trpo_paths', [True])
    vg.add('label_with_variation', [False])

    vg.add('append_extra_info', [True])
    vg.add('append_transformed_obs', [True])
    # baseline
    vg.add('baseline', ['g_mlp'])

    # vg.add('unif_goals', [True])  # put False for fixing the goal below!
    # vg.add('final_goal', [(0, 4)])

    # sampling params
    vg.add('horizon', [500])
    vg.add('outer_iters', [1000])
    vg.add('inner_iters', [5])
    vg.add('pg_batch_size', [100000])
    # policy initialization
    vg.add('output_gain', [1])
    vg.add('policy_init_std', [1])
    vg.add('learn_std', [True])
    vg.add('adaptive_std', [False])

    vg.add('seed', range(4000, 5000, 100))

    # # gan_configs
    # vg.add('GAN_batch_size', [128])  # proble with repeated name!!
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

    # Launching
    print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format(exp_prefix, vg.size))
    print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                                   config.AWS_SPOT_PRICE, n_parallel),
          *subnets)

    for vv in vg.variants():
        if mode in ['ec2', 'local_docker']:
            # choose subnet
            subnet = random.choice(subnets)
            config.AWS_REGION_NAME = subnet[:-1]
            config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[
                config.AWS_REGION_NAME]
            config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[
                config.AWS_REGION_NAME]
            config.AWS_SECURITY_GROUP_IDS = \
                config.ALL_REGION_AWS_SECURITY_GROUP_IDS[
                    config.AWS_REGION_NAME]
            config.AWS_NETWORK_INTERFACES = [
                dict(
                    SubnetId=config.ALL_SUBNET_INFO[subnet]["SubnetID"],
                    Groups=config.AWS_SECURITY_GROUP_IDS,
                    DeviceIndex=0,
                    AssociatePublicIpAddress=True,
                )
            ]

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
            )
