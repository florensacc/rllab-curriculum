"""
SnakeGather: find good size to compare agains baseline
"""

# imports -----------------------------------------------------
from rllab.config_personal import *
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.misc.nb_utils import ExperimentDatabase
from sandbox.carlos_snn.hallucinators.prior_hallucinator import PriorHallucinator
from sandbox.carlos_snn.old_my_snn.s_mlp_policy import GaussianMLPPolicy_snn
from sandbox.carlos_snn.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.carlos_snn.old_my_snn.trpo_snn import TRPO_snn
from rllab.algos.trpo import TRPO
from sandbox.carlos_snn.regressors.latent_regressor import Latent_regressor
import sys
import os
import math

# new things
from sandbox.carlos_snn.bonus_evaluators.grid_bonus_evaluator import GridBonusEvaluator

from sandbox.carlos_snn.envs.mujoco.maze.swimmer_maze_env import SwimmerMazeEnv
from sandbox.carlos_snn.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
from sandbox.carlos_snn.envs.mujoco.maze.snake_maze_env import SnakeMazeEnv
from sandbox.carlos_snn.envs.mujoco.snake_env import SnakeEnv
from sandbox.carlos_snn.envs.mujoco.gather.snake_gather_env import SnakeGatherEnv

from rllab.envs.normalized_env import normalize
from sandbox.carlos_snn.envs.hierarchized_snn_env import hierarchize_snn

stub(globals())

# exp setup --------------------------------------------------------
mode = "ec2"
ec2_instance = "c4.4xlarge"
# subnets =[
#     "us-west-1b"
# ]
# subnet = "us-west-1b"
info_instance = INSTANCE_TYPE_INFO[ec2_instance]
n_parallel = int(info_instance['vCPU']/2.)
spot_price = info_instance['price']

# for subnet in subnets:
aws_config = dict(
    # image_id=AWS_IMAGE_ID,
    instance_type=ec2_instance,
    # key_name=ALL_REGION_AWS_KEY_NAMES[subnet[:-1]],
    spot_price=str(spot_price),
    # security_group_ids=ALL_REGION_AWS_SECURITY_GROUP_IDS[subnet[:-1]],
)

exp_dir = 'data_upload/egoSnake64-snn/'
for dir in os.listdir(exp_dir):
    if 'Figure' not in dir and os.path.isfile(os.path.join(exp_dir, dir, 'params.pkl')):
        pkl_path = os.path.join(exp_dir, dir, 'params.pkl')
        print("hier for : ", pkl_path)

        for time_step_agg in [10, 50, 100]:

            for activity_range in [6, 10, 15]:
                inner_env = normalize(SnakeGatherEnv(activity_range=activity_range, sensor_range=activity_range,
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

                algo = TRPO_snn(
                    env=env,
                    policy=policy,
                    baseline=baseline,
                    self_normalize=True,
                    log_deterministic=True,
                    # reward_coef=reward_coef,
                    # bonus_evaluator=bonus_evaluators,
                    # reward_coef_bonus=reward_coef_bonus,
                    batch_size=5e5 / time_step_agg,
                    whole_paths=True,
                    max_path_length=5e3 / time_step_agg * activity_range / 6.,  # correct for larger envs
                    n_itr=2000,
                    discount=0.99,
                    step_size=0.01,
                )

                for s in [0]:  # range(10, 110, 10):  # [10, 20, 30, 40, 50]:
                    exp_prefix = 'hier-snn-egoSnake-gather'
                    exp_name = exp_prefix + '{}range_{}agg_{}pl_PRE{}_{}'.format(activity_range,
                                                                                 time_step_agg, int(
                            5e3 / time_step_agg * activity_range / 6.),
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
