'''
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
from rllab.config_personal import *
import math

# from rllab.envs.mujoco.swimmer_env import SwimmerEnv
# from sandbox.carlos_snn.envs.mujoco.snake_env import SnakeEnv
from sandbox.carlos_snn.envs.mujoco.maze.snake_maze_env import SnakeMazeEnv
from sandbox.carlos_snn.envs.mujoco.gather.snake_gather_env import SnakeGatherEnv

# from rllab.envs.mujoco.maze.ant_maze_env import AntMazeEnv
# from sandbox.carlos_snn.envs.mujoco.swimmer_env import SwimmerEnv
# from rllab.envs.mujoco.maze.swimmer_maze_env import SwimmerMazeEnv
# from sandbox.carlos_snn.envs.mujoco.maze.swimmer_maze_env import SwimmerMazeEnv
# from sandbox.carlos_snn.envs.mujoco.gather.gather_env import GatherEnv

stub(globals())

# exp setup --------------------------------------------------------
mode = "ec2"
ec2_instance = "m4.10xlarge"
# subnets =[
#     "us-west-1b"
# ]
# subnet = "us-west-1b"
info_instance = INSTANCE_TYPE_INFO[ec2_instance]
n_parallel = int(info_instance['vCPU'] / 2.)
# n_parallel = 4
spot_price = info_instance['price']

# for subnet in subnets:
aws_config = dict(
    # image_id=AWS_IMAGE_ID,
    instance_type=ec2_instance,
    # key_name=ALL_REGION_AWS_KEY_NAMES[subnet[:-1]],
    spot_price=str(spot_price),
    # security_group_ids=ALL_REGION_AWS_SECURITY_GROUP_IDS[subnet[:-1]],
)

for activity_range in [6, 10, 15]:
    env = normalize(SnakeGatherEnv(activity_range=activity_range, sensor_range=activity_range,
                                   sensor_span=math.pi * 2, ego_obs=True,
                                   coef_inner_rew=1))
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
        self_normalize=True,
        batch_size=5e5,
        whole_paths=True,
        max_path_length=5e3 * activity_range / 6.,  # correct for larger envs
        n_itr=2000,
        discount=0.99,
        step_size=0.01,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )

    for s in range(0, 50, 10):
        exp_prefix = 'trpo-egoSnake-gather'
        exp_name = exp_prefix + '{}scale_{}pl_{}'.format(activity_range,
                                                          int(5e3 * activity_range / 6.), s)
        run_experiment_lite(
            algo.train(),
            # where to launch the instances
            mode=mode,
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

