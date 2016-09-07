import os
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.hopper_env import HopperEnv
os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv

from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from rllab import config

stub(globals())

# Param ranges
seeds = list(range(10))

for seed in seeds:

    mdp_class = HopperEnv
    mdp = NormalizedEnv(env=mdp_class())

    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(64, 32),
    )

    baseline = GaussianMLPBaseline(
        mdp.spec,
        regressor_args=dict(hidden_sizes=(64, 32)),
    )

    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=1000,
        whole_paths=True,
        max_path_length=100,
        n_itr=5000,
        step_size=0.01,
        subsample_factor=1.0,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix=config.EXP_PREFIX + "_" + "hopper",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="lab_kube",
        dry=False,
    )
