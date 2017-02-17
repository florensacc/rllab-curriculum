'''
Launched Wednesday 11th. It was wrong: the normalize should have been on the inner env only!
'''
import math
import os

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.carlos_snn.envs.hierarchized_multiPol_env import hierarchize_multi
from sandbox.carlos_snn.envs.mujoco.gather.snake_gather_env import SnakeGatherEnv
from sandbox.carlos_snn.policies.categorical_mlp_policy import CategoricalMLPPolicy

stub(globals())

mesh_density = 5

for i in [1, 2, 3]:
    exp_dir = 'data_upload/egoSnake64-trpo-set1/'.format(i)
    pkl_paths = []
    json_paths = []
    for dir in os.listdir(exp_dir):
        pkl_path = os.path.join(exp_dir, dir, 'params.pkl')
        json_path = os.path.join(exp_dir, dir, 'params.json')
        if 'Figure' not in dir and os.path.isfile(pkl_path) and os.path.isfile(json_path):
            pkl_paths.append(pkl_path)
            json_paths.append(json_path)
            print("adding : ", pkl_paths[-1])

    for reward_coef in [0]:
        for time_step_agg in [1, 5, 10, 100]:
            for activity_range in [6, 10, 15]:
                inner_env = SnakeGatherEnv(activity_range=activity_range, sensor_range=activity_range,
                                           sensor_span=math.pi * 2, ego_obs=True)
                env = normalize(hierarchize_multi(inner_env, time_steps_agg=time_step_agg,
                                                  pkl_paths=pkl_paths, json_paths=json_paths
                                                  # animate=True
                                                  ))

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
                    log_deterministic=True,
                    reward_coef=reward_coef,
                    # bonus_evaluator=bonus_evaluators,
                    # reward_coef_bonus=reward_coef_bonus,
                    batch_size=1e5 / time_step_agg,
                    whole_paths=True,
                    max_path_length=int(5e3 / time_step_agg),
                    n_itr=2000,
                    discount=0.99,
                    step_size=0.01,
                )

                for s in [0, 100]:  # range(10, 110, 10):  # [10, 20, 30, 40, 50]:
                    exp_prefix = 'hier-multi-egoSnake-gather'
                    exp_name = exp_prefix + '_{}agg_{}pl_{}rewcoef_{}mesh_PREpostNIPS{}_{}'.format(
                        time_step_agg, int(5e3 / time_step_agg),
                        reward_coef, mesh_density, i, s)

                    run_experiment_lite(
                        stub_method_call=algo.train(),
                        mode='ec2',
                        pre_commands=['pip install --upgrade pip',
                                      'pip install --upgrade theano',
                                      ],
                        # pre_commands=['conda install -f mkl -n rllab3 -y'],
                        # Number of parallel workers for sampling
                        n_parallel=4,
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
