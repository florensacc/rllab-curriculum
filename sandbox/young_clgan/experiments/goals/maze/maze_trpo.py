import os

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import sys
from rllab.misc.instrument import run_experiment_lite
from rllab.misc.instrument import VariantGenerator
from sandbox.carlos_snn.autoclone import autoclone
from rllab import config

from sandbox.young_clgan.experiments.carlos_exps.goals.maze.maze_trpo_algo import run_task

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
        'ap-south-1a', 'us-east-2a', 'us-east-2b', 'us-east-2c', 'ap-south-1b', 'ap-northeast-2a', 'ap-northeast-2c',
        'us-east-1b', 'us-east-1a', 'us-east-1d', 'us-east-1e'
    ]
    ec2_instance = args.type if args.type else 'c4.2xlarge'
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
        n_parallel = 4
        # n_parallel = multiprocessing.cpu_count()

    exp_prefix = 'goal-trpo-maze3'

    vg = VariantGenerator()
    vg.add('horizon', [400])
    vg.add('goal_size', [2])  # this is the ultimate goal we care about: getting the pendulum upright
    vg.add('goal_range', [5])  # this will be used also as bound of the state_space
    vg.add('goal_center', [(2, 2)])
    vg.add('unif_goals', [False])
    vg.add('final_goal', [(-0.8, 4)])
    vg.add('goal_noise_level', [0.5])  # ???
    vg.add('terminal_eps', [0.3])
    vg.add('only_feasible', [True])
    vg.add('indicator_reward', [True])
    vg.add('min_reward', lambda indicator_reward: [10] if indicator_reward else [
        5])  # now running it with only the terminal reward of 1!
    vg.add('max_reward',
           lambda indicator_reward, terminal_eps: [900 * terminal_eps] if indicator_reward else [6e3])
    # vg.add('improvement_threshold', [10])  # is this based on the reward, now discounted success rate --> push for fast
    vg.add('outer_iters', [400])
    vg.add('inner_iters', [5])
    vg.add('pg_batch_size', [20000])
    vg.add('discount', [0.998])
    vg.add('gae_lambda', [0.995])

    vg.add('seed', range(50, 100, 20))
    vg.add('distance_metric', ['L2'])

    # policy initialization
    vg.add('output_gain', [1])
    vg.add('policy_init_std', [1])
    vg.add('learn_std', [True])
    vg.add('adaptive_std', [False])

    # Launch
    print("\n" + "**********"*10 + "\nexp_prefix: {}\nvariants: {}".format(exp_prefix, vg.size))
    print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                                   config.AWS_SPOT_PRICE,  n_parallel),
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
                    'pip install multiprocessing_on_dill',
                    'pip install scikit-image',
                    'conda install numpy -n rllab3 -y',
                ],
            )
            if mode == 'local_docker':
                sys.exit()
        else:
            run_experiment_lite(
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
