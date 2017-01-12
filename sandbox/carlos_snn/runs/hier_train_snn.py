"""
SnakeGather: find good size to compare agains baseline
"""

# imports -----------------------------------------------------
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
subnet = "us-west-1b"

n_parallel = 1

exp_dir = 'data_upload/egoSnake64-snn/'
for dir in os.listdir(exp_dir):
    if 'Figure' not in dir and os.path.isfile(os.path.join(exp_dir, dir, 'params.pkl')):
        pkl_path = os.path.join(exp_dir, dir, 'params.pkl')
        print("hier for : ", pkl_path)

        for time_step_agg in [500]:  # [1, 5, 10, 100]:

            # inner_env = normalize(SwimmerMazeEnv(maze_id=9, sensor_span=2*math.pi, goal_rew=1e4, ego_obs=True))
            # inner_env = normalize(SwimmerGatherEnv(sensor_span=math.pi*2, ego_obs=True))
            # inner_env = normalize(SnakeMazeEnv(maze_id=0, maze_size_scaling=6,
            #                       sensor_span=2*math.pi, goal_rew=5e3, ego_obs=True))
            # inner_env = normalize(SnakeEnv(ego_obs=True))
            inner_env = normalize(SnakeGatherEnv(activity_range=10, sensor_span=math.pi * 2, ego_obs=True))
            # env = normalize(hierarchize_snn(inner_env, time_steps_agg=time_step_agg, pkl_path=pkl_path,
            #                       # animate=True
            #                       ))
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
                batch_size=5e4 / time_step_agg,
                whole_paths=True,
                max_path_length=5e2 / time_step_agg,
                n_itr=2000,
                discount=0.99,
                step_size=0.01,
            )

            for s in [0, 100]:  # range(10, 110, 10):  # [10, 20, 30, 40, 50]:
                exp_prefix = 'Atest4-hier-snn-egoSnake'
                exp_name = exp_prefix + '_{}agg_{}pl_{}rewcoef_{}mesh_PRE{}_{}'.format(
                    time_step_agg, int(5e3 / time_step_agg),
                    # reward_coef, mesh_density,
                    pkl_path.split('/')[-2], s)

                run_experiment_lite(
                    stub_method_call=algo.train(),
                    mode='local',
                    pre_commands=['pip install --upgrade pip',
                                  'pip install --upgrade theano',
                                  ],
                    # pre_commands=['conda install -f mkl -n rllab3 -y'],
                    # Number of parallel workers for sampling
                    n_parallel=1,
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
