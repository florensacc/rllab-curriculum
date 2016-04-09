import os
os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

# Param ranges
seeds = range(2)

for seed in seeds:

    mdp_class = CartpoleEnv
    mdp = NormalizedEnv(env=mdp_class())

    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(32,),
    )

    baseline = LinearFeatureBaseline(
        mdp.spec
    )

    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=1000,
        whole_paths=False,
        max_path_length=100,
        n_itr=100,
        step_size=0.01,
        subsample_factor=1.0,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="cartpole",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="local_docker",
        dry=False,
    )
