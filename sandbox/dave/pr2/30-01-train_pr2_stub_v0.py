"""
Mon Jan 30 00:05:46 2017: _v0
"""
import multiprocessing

from sandbox.dave.pr2.action_limiter import FixedActionLimiter
from sandbox.dave.rllab.algos.trpo import TRPO
from sandbox.dave.rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from rllab.envs.normalized_env import normalize
from sandbox.dave.rllab.goal_generators.pr2_goal_generators import PR2CrownGoalGeneratorSmall #PR2CrownGoalGeneratorSmall
from sandbox.dave.rllab.lego_generators.pr2_lego_generators import PR2LegoBoxBlockGeneratorSmall #PR2LegoBoxBlockGeneratorSmall #PR2LegoBoxBlockGeneratorSmall #PR2LegoFixedBlockGenerator
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.dave.rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from sandbox.dave.rllab.policies.gaussian_mlp_policy_tanh import GaussianMLPPolicy
from rllab.misc.instrument import VariantGenerator, variant


from sandbox.dave.rllab.envs.mujoco.pr2_env_lego import Pr2EnvLego
# from sandbox.dave.rllab.envs.mujoco.pr2_env_lego_position import Pr2EnvLego
# from sandbox.dave.rllab.envs.mujoco.pr2_env_lego_hand import Pr2EnvLego
# from sandbox.dave.rllab.envs.mujoco.pr2_env_reach import Pr2EnvLego
# from sandbox.dave.rllab.envs.mujoco.pr2_env_lego_position_different_objects import Pr2EnvLego
from rllab import config
from sandbox.carlos_snn.autoclone import autoclone
import os
import random
import argparse


stub(globals())

train_goal_generator = PR2CrownGoalGeneratorSmall()
action_limiter = FixedActionLimiter()



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
        'us-east-2b', 'us-east-2a', 'us-east-2c', 'eu-west-1a', 'us-east-1b', 'us-east-1d', 'us-east-1a', 'us-east-1e',
        'eu-west-1c', 'ap-southeast-2b'
    ]

    ec2_instance = args.type if args.type else 'm4.16xlarge'

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

    print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                                   config.AWS_SPOT_PRICE, n_parallel),
          *subnets)

    seeds = range(50, 300, 50)  # [1, 11, 21, 31, 41]
    # seeds = [11, 21, 31, 41]
    # seeds = [11, 21, 31, 41]
    # std = [0.05, 0.1]
    # num_actions = [2, 5, 7, 10, 15, 20]
    # num_actions = [1]
    # taus = [1000, 2000]
    # taus = [0.9, 0.95, 0.995]
    # taus = [1]
    bool_tip = [0, 1]
    for t in bool_tip:
        for s in seeds:
            env = normalize(Pr2EnvLego(
                goal_generator=train_goal_generator,
                lego_generator=PR2LegoBoxBlockGeneratorSmall(),
                # action_limiter=action_limiter,
                max_action=1,
                pos_normal_sample=True,
                qvel_init_std=0.01,
                pos_normal_sample_std=.01,  #0.5
                # use_depth=True,
                # use_vision=True,
                allow_random_restarts=True,
                tip=t,
                # tau=t,
                # crop=True,
                # allow_random_vel_restarts=True,
                ))

            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                # The neural network policy should have n hidden layers, each with k hidden units.
                hidden_sizes=(64, 64, 64),
                # init_std=0.1,
                # output_gain=0.1,
                # beta=0.05,
                # pkl_path= "upload/fixed-arm-position-ctrl-tip-no-random-restarts/fixed-arm-position-ctrl-tip-no-random-restarts1/params.pkl"
                # json_path="/home/ignasi/GitRepos/rllab-goals/data/local/train-Lego/rand_init_angle_reward_shaping_continuex2_2016_10_17_12_48_20_0001/params.json",
                )

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=50000,
                max_path_length=150,  #100
                n_itr=10000, #50000
                discount=0.95,
                gae_lambda=0.98,
                step_size=0.01,
                goal_generator=train_goal_generator,
                action_limiter=None,
                optimizer_args={'subsample_factor': 0.1},
                # discount_weights={'angle': 0.1, 'tip': .1},
                # plot=True,
                # Uncomment both lines (this and the plot parameter below) to enable plotting
                )

            # algo.train()

        ##lambda exp: exp.params['exp_name'].split('_')[-1][:2]
            exp_prefix = "train-Lego/RSS/rewards"
            if t:
                exp_name="r_tip_" + str(s)
            else:
                exp_name = "r_base_" + str(s)

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
                    n_parallel=n_parallel,
                    # Only keep the snapshot parameters for the last iteration
                    snapshot_mode="last",
                    seed=s,
                    # plot=True,
                    exp_prefix=exp_prefix,
                    exp_name=exp_name,
                )
                sys.exit()
