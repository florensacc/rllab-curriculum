"""
Fri Jan 27 15:07:20 2017: _v0: tests for AntFollow with plain TRPO
"""
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
from sandbox.carlos_snn.envs.mujoco.follow.ant_follow_env import AntFollowEnv

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

    # setup ec2
    subnets = [
        'us-east-2c', 'us-east-2b', 'us-west-1b', 'ap-south-1a', 'us-east-1e', 'us-east-1d', 'us-east-1b',
        'ap-northeast-2c', 'ap-south-1b', 'us-east-2a', 'us-east-1a'
    ]

    ec2_instance = args.type if args.type else 'c4.8xlarge'

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

    print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                                   config.AWS_SPOT_PRICE, n_parallel),
          *subnets)

    sensor_range = 10

    for displ_std in [0.1, 0.3, 0.5]:
        for sensor_span in [math.pi, 2*math.pi]:
            for goal_dist_rew in [0, 1e-5]:
                for goal_vector_obs in [True, False]:

                    env = normalize(AntFollowEnv(sensor_span=sensor_span, sensor_range=sensor_range,
                                                 goal_dist_rew=goal_dist_rew,
                                                 goal_vector_obs=goal_vector_obs, ego_obs=True,
                                                 ))

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
                        batch_size=1e5,
                        whole_paths=True,
                        max_path_length=5e3,
                        n_itr=2000,
                        discount=0.99,
                        step_size=0.01,
                    )

                    for s in range(0, 40, 10):
                        exp_prefix = 'trpo-AntFollow'
                        exp_name = exp_prefix + '_{}displStd_{}sensSpan_{}goalDistRew_{}goalVectObs_{}'.format(
                                                ''.join(str(displ_std).split('.')), int(sensor_span),
                                                ''.join(str(goal_dist_rew).split('.')), str(goal_vector_obs), s)
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
                                    "pip install --upgrade pip",
                                    "pip install --upgrade theano"
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

