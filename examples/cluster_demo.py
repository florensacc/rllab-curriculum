import os
import random

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab import config
import sys


def run_task(v):
    env = normalize(CartpoleEnv())

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=40,
        discount=0.99,
        step_size=v["step_size"],
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()


for step_size in [0.01, 0.05, 0.1]:
    for seed in [1, 11, 21, 31, 41]:
        subnets = [
             "us-east-1d",
        ]
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
            run_task,
            exp_prefix="first_exp",
            # Number of parallel workers for sampling
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=seed,
            mode="ec2",
            # mode="local_docker",
            variant=dict(step_size=step_size, seed=seed)
            # plot=True,
            # terminate_machine=False,
        )
        sys.exit()
