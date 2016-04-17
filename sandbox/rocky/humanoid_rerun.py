from rllab.algos.cem import CEM
from rllab.algos.cma_es import CMAES
from rllab.algos.reps import REPS
from rllab.algos.vpg import VPG
from rllab.algos.tnpg import TNPG
from rllab.algos.trpo import TRPO
from rllab.algos.erwr import ERWR

from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
import lasagne.nonlinearities
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from rllab.misc.instrument import stub, run_experiment_lite
import sys

stub(globals())

N_ITR = 500
HORIZON = 500
BATCH_SIZE = 50000
DISCOUNT = 0.99

for seed in [28, 43, 68, 103, 148]:

    for env in map(normalize, [SwimmerEnv(), HalfCheetahEnv(), Walker2DEnv(), HumanoidEnv(), SimpleHumanoidEnv()]):
        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(100, 50, 25),
                                   hidden_nonlinearity=lasagne.nonlinearities.tanh)
        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algos = [
            CMAES(
                env=env,
                policy=policy,
                n_itr=N_ITR,
                max_path_length=HORIZON,
                discount=DISCOUNT,
                batch_size=BATCH_SIZE,
                sigma0=1e-1,
            ),
            CEM(
                env=env,
                policy=policy,
                n_itr=N_ITR,
                max_path_length=HORIZON,
                discount=DISCOUNT,
                batch_size=BATCH_SIZE,
                extra_std=1e-3,
            ),
            REPS(
                env=env,
                policy=policy,
                baseline=baseline,
                n_itr=N_ITR,
                epsilon=8e-1,
                batch_size=BATCH_SIZE,
                max_path_length=HORIZON,
                discount=DISCOUNT,
            ),
            REPS(
                env=env,
                policy=policy,
                baseline=baseline,
                n_itr=N_ITR,
                epsilon=3e-1,
                batch_size=BATCH_SIZE,
                max_path_length=HORIZON,
                discount=DISCOUNT,
            ),
            TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                n_itr=N_ITR,
                batch_size=BATCH_SIZE,
                max_path_length=HORIZON,
                discount=DISCOUNT,
                step_size=3e-1,
            ),
            TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                n_itr=N_ITR,
                batch_size=BATCH_SIZE,
                max_path_length=HORIZON,
                discount=DISCOUNT,
                step_size=2e-1,
            ),
            TNPG(
                env=env,
                policy=policy,
                baseline=baseline,
                n_itr=N_ITR,
                batch_size=BATCH_SIZE,
                max_path_length=HORIZON,
                discount=DISCOUNT,
                step_size=3e-1,
            ),
            TNPG(
                env=env,
                policy=policy,
                baseline=baseline,
                n_itr=N_ITR,
                batch_size=BATCH_SIZE,
                max_path_length=HORIZON,
                discount=DISCOUNT,
                step_size=1e-1,
            ),
            ERWR(
                env=env,
                policy=policy,
                baseline=baseline,
                n_itr=N_ITR,
                batch_size=BATCH_SIZE,
                max_path_length=HORIZON,
                discount=DISCOUNT,
            ),
            VPG(
                env=env,
                policy=policy,
                baseline=baseline,
                n_itr=N_ITR,
                batch_size=BATCH_SIZE,
                max_path_length=HORIZON,
                discount=DISCOUNT,
                optimizer_args=dict(learning_rate=5e-3),
            ),
            VPG(
                env=env,
                policy=policy,
                baseline=baseline,
                n_itr=N_ITR,
                batch_size=BATCH_SIZE,
                max_path_length=HORIZON,
                discount=DISCOUNT,
                optimizer_args=dict(learning_rate=1e-2),
            ),
        ]

        for algo in algos:
            run_experiment_lite(
                algo.train(),
                mode="ec2",
                exp_prefix="icml_all_tanh_again",
                n_parallel=4,
                snapshot_mode="last",
                seed=seed,
            )
