'''
01/14/2017
Run in the setting of the main part of the paper, to check how does TRPO does!
train baseline -- Add the coef inner rew of 1e-4 (before the order of maginitude of the inner rew was maybe too large)
'''
# from rllab.sampler import parallel_sampler
# parallel_sampler.initialize(n_parallel=2)
# parallel_sampler.set_seed(1)

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.config_personal import *
import math

from sandbox.carlos_snn.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv

stub(globals())

# exp setup --------------------------------------------------------
mode = "ec2"
ec2_instance = "m4.2xlarge"
# subnets =[
#     "us-west-1b"
# ]
# subnet = "us-west-1b"
info_instance = INSTANCE_TYPE_INFO[ec2_instance]
n_parallel = int(info_instance['vCPU'] / 2)
# n_parallel = 1
spot_price = str(info_instance['price'])

# for subnet in subnets:
aws_config = dict(
    # image_id=AWS_IMAGE_ID,
    instance_type=ec2_instance,
    # key_name=ALL_REGION_AWS_KEY_NAMES[subnet[:-1]],
    spot_price=str(spot_price),
    # security_group_ids=ALL_REGION_AWS_SECURITY_GROUP_IDS[subnet[:-1]],
)

for activity_range in [6]:
    for coef_inner_rew in [0, 1e-4]:
        env = normalize(SwimmerGatherEnv(sensor_span=math.pi * 2, ego_obs=True, coef_inner_rew=coef_inner_rew))

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
            max_path_length=int(5e3 * activity_range / 6.),  # correct for larger envs
            n_itr=2000,
            discount=0.99,
            step_size=0.01,
            # Uncomment both lines (this and the plot parameter below) to enable plotting
            # plot=True,
        )

        for s in range(0, 50, 10):
            exp_prefix = 'trpo-egoSwimmer-gather'
            exp_name = exp_prefix + '{}in_{}pl_{}'.format(''.join(str(coef_inner_rew).split('.')), 5e3, s)
            run_experiment_lite(
                algo.train(),
                # where to launch the instances
                mode=mode,
                aws_config=aws_config,
                pre_commands=['pip install --upgrade pip',
                              'pip install --upgrade theano',
                              ],
                # Number of parallel workers for sampling
                n_parallel=n_parallel,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                ## !!!
                sync_s3_pkl=True,
                sync_s3_png=True,
                # Specifies the seed for the experiment. If this is not provided, a random seed
                # will be used
                seed=s,
                # plot=True,
                exp_prefix=exp_prefix,
                exp_name=exp_name,
            )
