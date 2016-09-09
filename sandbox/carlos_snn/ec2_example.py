


from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
import sys

stub(globals())

for alive_coeff in [ 1.0]:
    # for seed in [1, 2, 3, 4, 5]:
    seed =1
    env = HopperEnv(alive_coeff=alive_coeff)
    policy = GaussianMLPPolicy(env_spec=env.spec)
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        step_size=0.01,
        n_itr=500,
        batch_size=10000,
    )

    run_experiment_lite(
        algo.train(),
        mode="ec2",
        n_parallel=1,
        snapshot_mode="last",
        exp_prefix="charlio_trpo_hopper",
        seed=seed,
        terminate_machine=True, # dangerous to have False!!
    )
    sys.exit(0)