'''
train baseline
'''
# from rllab.sampler import parallel_sampler
# parallel_sampler.initialize(n_parallel=2)
# parallel_sampler.set_seed(1)

import math

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.config_personal import *
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.carlos_snn.envs.mujoco.maze.snake_maze_env import SnakeMazeEnv

# from rllab.envs.mujoco.maze.ant_maze_env import AntMazeEnv
# from sandbox.carlos_snn.envs.mujoco.swimmer_env import SwimmerEnv
# from rllab.envs.mujoco.maze.swimmer_maze_env import SwimmerMazeEnv
# from sandbox.carlos_snn.envs.mujoco.maze.swimmer_maze_env import SwimmerMazeEnv
# from sandbox.carlos_snn.envs.mujoco.gather.gather_env import GatherEnv

stub(globals())

# env = normalize(SwimmerEnv(ego_obs=True))
# env = normalize(SnakeMazeEnv(maze_id=3, sensor_span=2*math.pi))  #, ego_obs=True))
# env = normalize(SnakeMazeEnv(maze_id=3, sensor_span=2*math.pi, ego_obs=True))
# env = SwimmerMazeEnv(sensor_span=math.pi*2, ctrl_cost_coeff=1)

# exp setup --------------------------------------------------------
mode = "ec2"
ec2_instance = "m4.4xlarge"
# subnets =[
#     "us-west-1b"
# ]
# subnet = "us-west-1b"
info_instance = INSTANCE_TYPE_INFO[ec2_instance]
n_parallel = int(info_instance['vCPU'] / 2.)
spot_price = str(info_instance['price'])

# for subnet in subnets:
aws_config = dict(
    # image_id=AWS_IMAGE_ID,
    instance_type=ec2_instance,
    # key_name=ALL_REGION_AWS_KEY_NAMES[subnet[:-1]],
    spot_price=str(spot_price),
    # security_group_ids=ALL_REGION_AWS_SECURITY_GROUP_IDS[subnet[:-1]],
)

for maze_size_scaling in [6, 10, 15]:
    env = normalize(SnakeMazeEnv(maze_id=0, sensor_span=math.pi * 2, ego_obs=True,
                                 maze_size_scaling=maze_size_scaling,
                                 coef_inner_rew=1, goal_rew=1e4))

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
        max_path_length=1e4 * maze_size_scaling / 4.,
        n_itr=200,
        discount=0.99,
        step_size=0.01,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )

    for s in range(0, 50, 10):
        exp_prefix = 'trpo-egoSnake-maze0'
        exp_name = exp_prefix + '{}scale_{}pl__{}'.format(maze_size_scaling,
                                                          int(1e4 * maze_size_scaling / 4.), s)
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

        # print("about to train the algo")
        # algo.train()
