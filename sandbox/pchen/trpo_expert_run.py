import os

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.normalized_env import NormalizedEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

mdp_classes = [
    Walker2DEnv,
    SwimmerEnv,
    SimpleHumanoidEnv,
    HalfCheetahEnv,
    HopperEnv,
]

for mdp_class in mdp_classes:
    mdp = NormalizedEnv(env=mdp_class())
    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(100, 50, 25),
        # nonlinearity='lasagne.nonlinearities.rectified',
    )
    baseline = LinearFeatureBaseline(
        mdp.spec
    )
    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=50000,
        whole_paths=True,
        max_path_length=500,
        n_itr=2,
        discount=0.99,
        step_size=5e-2,
        store_paths=True,
    )
    for seed in [1, 3]:
        run_experiment_lite(
            algo.train(),
            exp_prefix="trpo_expert_run",
            n_parallel=4,
            snapshot_mode="all",
            seed=1,
            dry=False,
            mode="lab_kube",
        )
        break
    break
