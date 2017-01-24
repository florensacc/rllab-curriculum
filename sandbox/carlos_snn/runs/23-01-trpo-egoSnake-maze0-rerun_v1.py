"""
<<<<<<< HEAD
Mon Jan 23 23:49:34 2017: _v1: this still didn't get through ec2!!?
=======
Mon Jan 23 23:49:34 2017: _v1
>>>>>>> 23-01-trpo-egoSnake-maze0-rerun_v1.py
Fri Jan 20 11:16:18 2017: _v0
"""
'''
01/14/2017
Rerun adjusting the goalReward
train baseline
'''
# from rllab.sampler import parallel_sampler
# parallel_sampler.initialize(n_parallel=2)
# parallel_sampler.set_seed(1)

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab import config
import math
import random
import argparse
import sys
from sandbox.carlos_snn.autoclone import autoclone


from sandbox.carlos_snn.envs.mujoco.maze.snake_maze_env import SnakeMazeEnv

stub(globals())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ec2', '-e', action='store_true', default=False, help="add flag to run in ec2")
    parser.add_argument('--clone', '-c', action='store_true', default=False, help="add flag to copy file and checkout current")
    parser.add_argument('--local_docker', '-d', action='store_true', default=False,
                        help="add flag to run in local dock")
    parser.add_argument('--type', '-t', type=str, default='', help='set instance type')
    parser.add_argument('--price', '-p', type=str, default='', help='set betting price')
    parser.add_argument('--subnet', '-sn', type=str, default='', help='set subnet like us-west-1a')
    parser.add_argument('--name', '-n', type=str, default='', help='set exp prefix name and new file name')
    args = parser.parse_args()

    if args.clone:
        autoclone.autoclone(__file__, args)
        sys.exit()

    # setup ec2
    subnets = [
        'us-east-2c', 'us-east-2b', 'us-east-2a'
    ]

    ec2_instance = args.type if args.type else 'c4.4xlarge'

    # configure instance
    info = config.INSTANCE_TYPE_INFO[ec2_instance]
    config.AWS_INSTANCE_TYPE = ec2_instance
    config.AWS_SPOT_PRICE = str(info["price"])
    n_parallel = int(info["vCPU"] / 2)  # make the default 4 if not using ec2

    if args.ec2:
        mode = 'ec2'
    elif args.local_docker:
        mode = 'local_docker'
        n_parallel = 1
    else:
        mode = 'local'

    print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                                   config.AWS_SPOT_PRICE, n_parallel),
          *subnets)

    goal_rew = 1e6
    for maze_size_scaling in [7]:
        env = normalize(SnakeMazeEnv(maze_id=0, sensor_span=math.pi * 2, ego_obs=True,
                                     maze_size_scaling=maze_size_scaling,
                                     coef_inner_rew=1, goal_rew=goal_rew))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=(64, 64)
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=1e6,
            whole_paths=True,
            max_path_length=1e4 * maze_size_scaling / 2.,
            n_itr=200,
            discount=0.99,
            step_size=0.01,
            # Uncomment both lines (this and the plot parameter below) to enable plotting
            # plot=True,
        )

        for s in range(0, 110, 10):
            exp_prefix = 'trpo-egoSnake-maze0-bis'
            exp_name = exp_prefix + '_{}goalRew_{}scale_{}pl_{}'.format(int(goal_rew), maze_size_scaling,
                                                              int(1e4 * maze_size_scaling / 2.), s)
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
                    stub_method_call=algo.train(),
                    mode=mode,
                    # Number of parallel workers for sampling
                    n_parallel=n_parallel,
                    # Only keep the snapshot parameters for the last iteration
                    snapshot_mode="last",
                    seed=s,
                    # plot=True,
                    exp_prefix=exp_prefix,
                    exp_name=exp_name,
                    sync_s3_pkl=True,
                    # for sync the pkl file also during the training
                    sync_s3_png=True,
                    # # use this ONLY with ec2 or local_docker!!!
                    pre_commands=[
                        "which conda",
                        "which python",
                        "conda list -n rllab3",
                        "conda install -f numpy -n rllab3 -y",
                    ],
                )
                if mode == 'local_docker':
                    sys.exit()
            else:
                run_experiment_lite(
                    stub_method_call=algo.train(),
                    mode='local',
                    n_parallel=1,
                    # Only keep the snapshot parameters for the last iteration
                    snapshot_mode="last",
                    seed=s,
                    # plot=True,
                    exp_prefix=exp_prefix,
                    exp_name=exp_name,
                )
                sys.exit()

