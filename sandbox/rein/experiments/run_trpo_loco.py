import os
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv

from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools

stub(globals())

# Param ranges
seeds = range(10)
mdp_classes = [Walker2DEnv]
mdps = [NormalizedEnv(env=mdp_class()) for mdp_class in mdp_classes]
param_cart_product = itertools.product(
   mdps, seeds 
)

for mdp, seed in param_cart_product:

    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(32,),
    )

    baseline = GaussianMLPBaseline(
        mdp.spec,
        regressor_args=dict(hidden_sizes=(32,)),
    )

    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=10000,
        whole_paths=True,
        max_path_length=500,
        n_itr=10000,
        step_size=0.001,
        subsample_factor=1.0,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-loco-v1",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="lab_kube",
        dry=False,
    )
