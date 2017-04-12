import argparse
import math
import os
import os.path as osp
import sys

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Symbols that need to be stubbed
from rllab.misc.instrument import run_experiment_lite
from rllab import config
from rllab.misc.instrument import VariantGenerator

from sandbox.carlos_snn.autoclone import autoclone

# from sandbox.young_clgan.lib.utils import initialize_parallel_sampler
# initialize_parallel_sampler()

from sandbox.young_clgan.experiments.carlos_exps.goals.point_trpo_algo import run_task

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]

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
        'us-east-2c', 'us-east-2a', 'us-east-2b', 'ap-southeast-2b', 'ap-southeast-1b', 'ap-northeast-1c', 'us-west-2b',
        'ap-northeast-2a', 'ap-southeast-2c'
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

    exp_prefix = 'goal-point-trpo-indicator-unifFeas'
    vg = VariantGenerator()

    vg.add('seed', range(30, 90, 20))
    # # GeneratorEnv params
    vg.add('goal_size', [6, 5, 4, 3, 2])  # this is the ultimate goal we care about: getting the pendulum upright
    vg.add('goal_range', [5])  # this will be used also as bound of the state_space
    vg.add('sample_unif_feas', [True])
    vg.add('reward_dist_threshold', lambda goal_size: [math.sqrt(goal_size) / math.sqrt(2) * 0.3])
    # vg.add('angle_idxs', [((0, 1),)]) # these are the idx of the obs corresponding to angles (here the first 2)
    vg.add('distance_metric', ['L2'])
    vg.add('terminal_bonus', [300])
    vg.add('terminal_eps', lambda reward_dist_threshold: [
        reward_dist_threshold])  # if hte terminal bonus is 0 it doesn't kill it! Just count how many reached center
    vg.add('state_bounds', lambda goal_range, goal_size, reward_dist_threshold:
    [(1, goal_range) + (0.3,) * (goal_size - 2) + (goal_range, ) * goal_size])
    #############################################
    vg.add('min_reward', lambda terminal_bonus: [terminal_bonus * 0.1])  # now running it with only the terminal reward of 1!
    vg.add('max_reward', lambda terminal_bonus: [terminal_bonus * 0.9])
    vg.add('horizon', [200])
    vg.add('outer_iters', [400])
    vg.add('inner_iters', [5])
    vg.add('pg_batch_size', [20000])
    # policy initialization
    vg.add('output_gain', [1])
    vg.add('policy_init_std', [1])

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
                sync_s3_pkl=True,
                # for sync the pkl file also during the training
                sync_s3_png=True,
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
