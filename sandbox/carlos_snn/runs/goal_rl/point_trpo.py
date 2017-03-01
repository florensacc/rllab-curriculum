import sys
import os
import os.path as osp
import argparse
import math
import random
import numpy as np

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Symbols that need to be stubbed
from rllab.algos.trpo import TRPO
from sandbox.carlos_snn.algos.trpo_goal import TRPOGoal
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
import rllab.misc.logger
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sampler.stateful_pool import singleton_pool
from rllab.envs.box2d.pendulum_env import PendulumEnv
from sandbox.carlos_snn.envs.point_env import PointEnv
from rllab import config
from rllab.misc.instrument import VariantGenerator, variant

from sandbox.young_clgan.lib.envs.base import GoalExplorationEnv, GoalIdxExplorationEnv
from sandbox.young_clgan.lib.envs.base import UniformGoalGenerator, FixedGoalGenerator
from sandbox.young_clgan.lib.goal import *
from sandbox.young_clgan.lib.logging import *
from sandbox.carlos_snn.autoclone import autoclone

# from sandbox.young_clgan.lib.utils import initialize_parallel_sampler
# initialize_parallel_sampler()

from sandbox.carlos_snn.runs.goal_rl.point_trpo2 import run_task

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
    print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                                   config.AWS_SPOT_PRICE, n_parallel),
          *subnets)

    exp_prefix = 'goal-point-trpo5'
    vg = VariantGenerator()

    vg.add('seed', range(10, 40, 10))
    # # GeneratorEnv params
    vg.add('goal_size', [6, 5, 4, 3, 2, 1])  # this is the ultimate goal we care about: getting the pendulum upright
    vg.add('goal_range', [5])  # this will be used also as bound of the state_space
    vg.add('reward_dist_threshold', lambda goal_size: [math.sqrt(goal_size) / math.sqrt(2) * 0.5])
    vg.add('state_bounds', lambda reward_dist_threshold, goal_range, goal_size:
    [(1, goal_range) + (reward_dist_threshold,) * (goal_size - 2) + (goal_range, ) * goal_size])
    # vg.add('angle_idxs', [((0, 1),)]) # these are the idx of the obs corresponding to angles (here the first 2)
    vg.add('distance_metric', ['L2'])
    vg.add('terminal_bonus', [0])
    vg.add('terminal_eps', lambda reward_dist_threshold: [
        reward_dist_threshold])  # if hte terminal bonus is 0 it doesn't kill it! Just count how many reached center
    #############################################
    vg.add('min_reward', [1])  # now running it with only the terminal reward of 1!
    vg.add('max_reward', [1e3])
    vg.add('horizon', [200])
    vg.add('outer_iters', [500])
    vg.add('inner_iters', [5])
    vg.add('pg_batch_size', [20000])
    # policy initialization
    vg.add('output_gain', [0.1])
    vg.add('policy_init_std', [0.1])

    # def run_task(v):
    #     # random.seed(v['seed'])
    #     # np.random.seed(v['seed'])
    #
    #     inner_env = normalize(PointEnv(dim=v['goal_size'], state_bounds=v['state_bounds']))
    #     goal_generator = UniformGoalGenerator(goal_size=v['goal_size'], bounds=[-1 * v['goal_range'] * np.ones(v['goal_size']),
    #                                                                             v['goal_range'] * np.ones(v['goal_size'])])
    #
    #     env = GoalIdxExplorationEnv(env=inner_env, goal_generator=goal_generator,
    #                                 idx=np.arange(v['goal_size']),
    #                                 reward_dist_threshold=v['reward_dist_threshold'],
    #                                 distance_metric=v['distance_metric'],
    #                                 terminal_eps=v['terminal_eps'], terminal_bonus=v['terminal_bonus'],
    #                                 )  # this goal_generator will be updated by a uniform after
    #
    #     policy = GaussianMLPPolicy(
    #         env_spec=env.spec,
    #         hidden_sizes=(32, 32),
    #         # Fix the variance since different goals will require different variances, making this parameter hard to learn.
    #         learn_std=False,
    #         init_std=0.1,
    #     )
    #
    #     baseline = LinearFeatureBaseline(env_spec=env.spec)
    #
    #     algo = TRPOGoal(
    #         env=env,
    #         policy=policy,
    #         baseline=baseline,
    #         batch_size=v['pg_batch_size'],
    #         max_path_length=v['horizon'],
    #         n_itr=v['n_itr'],
    #         discount=0.99,
    #         step_size=0.01,
    #         plot=False,
    #     )
    #
    #     algo.train()


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
