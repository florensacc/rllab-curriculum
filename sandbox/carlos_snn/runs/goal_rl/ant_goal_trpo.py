import sys
import os
import os.path as osp
import argparse
import random

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Symbols that need to be stubbed
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
import rllab.misc.logger
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sampler.stateful_pool import singleton_pool
from sandbox.carlos_snn.envs.mujoco.ant_env import AntEnv
from rllab import config

from sandbox.young_clgan.lib.envs.base import GoalIdxExplorationEnv
from sandbox.young_clgan.lib.envs.base import UniformGoalGenerator, FixedGoalGenerator, update_env_goal_generator
from sandbox.young_clgan.lib.goal import *
from sandbox.young_clgan.lib.logging import *
from sandbox.carlos_snn.autoclone import autoclone

# from sandbox.young_clgan.lib.utils import initialize_parallel_sampler
# initialize_parallel_sampler()

stub(globals())

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
        'us-east-2b', 'us-east-1a', 'us-east-1d', 'us-east-1b', 'us-east-1e', 'ap-south-1b', 'ap-south-1a', 'us-west-1a'
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
        n_parallel = 4
    else:
        mode = 'local'
        n_parallel = 1
    print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                                   config.AWS_SPOT_PRICE, n_parallel),
          *subnets)

    n_itr = 2000
    batch_size = 50000
    max_path_length = 500  # horizon for the robot

    for goal_reward in ['InverseDistance']:  # don't use the negative as Ant will kill itself!
        for inner_weight in [1, 0]:  # to get the ctrl, contact and survival cost (not the distance!)
            for goal_range in [2, 4]:
                inner_env = normalize(AntEnv(sparse=True))
                goal_generator = UniformGoalGenerator(goal_size=2, bound=goal_range)
                env = GoalIdxExplorationEnv(inner_env, goal_generator, goal_weight=1, goal_reward=goal_reward,
                                            inner_weight=inner_weight)  # this goal_generator will be updated by a uniform after

                policy = GaussianMLPPolicy(
                    env_spec=env.spec,
                    hidden_sizes=(64, 64),
                    # Fix the variance since different goals will require different variances, making this parameter hard to learn.
                    learn_std=False
                )

                baseline = LinearFeatureBaseline(env_spec=env.spec)

                algo = TRPO(
                    env=env,
                    policy=policy,
                    baseline=baseline,
                    batch_size=batch_size,
                    max_path_length=max_path_length,
                    n_itr=n_itr,
                    discount=0.99,
                    step_size=0.01,
                    plot=False,
                )

                for s in range(0, 20, 10):
                    exp_prefix = 'goal-ant-trpo'
                    exp_name = exp_prefix + '{}inRew_{}range_{}_{}s'.format(inner_weight, goal_range, goal_reward, s)
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
                            use_cloudpickle=False,
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
                            use_cloudpickle=False,
                            stub_method_call=algo.train(),
                            mode='local',
                            n_parallel=n_parallel,
                            # Only keep the snapshot parameters for the last iteration
                            snapshot_mode="last",
                            seed=s,
                            # plot=True,
                            exp_prefix=exp_prefix,
                            exp_name=exp_name,
                        )
