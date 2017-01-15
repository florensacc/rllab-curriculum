"""
01/15/2017 1:48 (the one from yesterday had the wrong bs and pl!!!
Run one more random seed of each
SnakeGather: find good size to compare agains baseline
"""

# imports -----------------------------------------------------
import math
import os

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.config_personal import *
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
# from sandbox.carlos_snn.algos.trpo_snn import TRPO_snn
from rllab.algos.trpo import TRPO
from sandbox.carlos_snn.envs.hierarchized_snn_env import hierarchize_snn
from sandbox.carlos_snn.envs.mujoco.maze.snake_maze_env import SnakeMazeEnv
from sandbox.carlos_snn.policies.categorical_mlp_policy import CategoricalMLPPolicy

stub(globals())

# exp setup --------------------------------------------------------
mode = "local_docker"
local_instance = "m4.16xlarge"
# subnets =[
#     "us-west-1b"
# ]
# subnet = "us-west-1b"
info_instance = INSTANCE_TYPE_INFO[local_instance]
n_parallel = int(info_instance['vCPU']/2.)
spot_price = info_instance['price']

# for subnet in subnets:
aws_config = dict(
    # image_id=AWS_IMAGE_ID,
    instance_type=local_instance,
    # key_name=ALL_REGION_AWS_KEY_NAMES[subnet[:-1]],
    spot_price=str(spot_price),
    # security_group_ids=ALL_REGION_AWS_SECURITY_GROUP_IDS[subnet[:-1]],
)

exp_dir = 'data_upload/egoSnake64-snn/'
for dir in os.listdir(exp_dir):
    if 'Figure' not in dir and os.path.isfile(os.path.join(exp_dir, dir, 'params.pkl')):
        pkl_path = os.path.join(exp_dir, dir, 'params.pkl')
        print("hier for : ", pkl_path)

        for maze_size_scaling in [7, 9]:

            for time_step_agg in [100, 500, 800]:

                inner_env = normalize(SnakeMazeEnv(maze_id=0, maze_size_scaling=maze_size_scaling,
                                                   sensor_span=math.pi * 2, ego_obs=True))
                env = hierarchize_snn(inner_env, time_steps_agg=time_step_agg, pkl_path=pkl_path,
                                      # animate=True,
                                      )

                policy = CategoricalMLPPolicy(
                    env_spec=env.spec,
                )

                baseline = LinearFeatureBaseline(env_spec=env.spec)

                # bonus_evaluators = [GridBonusEvaluator(mesh_density=mesh_density, visitation_bonus=1, snn_H_bonus=0)]
                # reward_coef_bonus = [reward_coef]

                algo = TRPO(
                    env=env,
                    policy=policy,
                    baseline=baseline,
                    self_normalize=True,
                    log_deterministic=True,
                    # reward_coef=reward_coef,
                    # bonus_evaluator=bonus_evaluators,
                    # reward_coef_bonus=reward_coef_bonus,
                    batch_size=1e6 / time_step_agg,
                    whole_paths=True,
                    max_path_length=1e4 / time_step_agg * maze_size_scaling / 2.,  # correct for larger envs
                    n_itr=200,
                    discount=0.99,
                    step_size=0.01,
                )

                for s in [10]:  # range(10, 110, 10):  # [10, 20, 30, 40, 50]:
                    exp_prefix = 'hier-snn-egoSnake-maze0'
                    exp_name = exp_prefix + '{}scale_{}agg_{}pl_PRE{}_{}'.format(maze_size_scaling,
                                                                                 time_step_agg, int(
                            1e4 / time_step_agg * maze_size_scaling / 2.),
                                                                                 pkl_path.split('/')[-2], s)

                    run_experiment_lite(
                        stub_method_call=algo.train(),
                        mode=mode,
                        aws_config=aws_config,
                        pre_commands=['pip install --upgrade pip',
                                      'pip install --upgrade theano',
                                      ],
                        # Number of parallel workers for sampling
                        n_parallel=n_parallel,
                        # n_parallel=1,
                        # Only keep the snapshot parameters for the last iteration
                        snapshot_mode="last",
                        # Specifies the seed for the experiment. If this is not provided, a random seed
                        # will be used
                        seed=s,
                        # plot=True,
                        # Save to data/local/exp_prefix/exp_name/
                        exp_prefix=exp_prefix,
                        exp_name=exp_name,
                        sync_s3_pkl=True,  # for sync the pkl file also during the training
                        sync_s3_png=True,
                        terminate_machine=True,  # dangerous to have False!
                    )
