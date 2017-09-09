from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite, VariantGenerator
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.ignasi.robust_disk.envs.disk_generate_states_env import DiskGenerateStatesEnv

"""
Allows for testing of Brownian motion.
Set init_state in reset to init_state = (0.387, 1.137, -2.028, -1.744, 2.029, -0.873, 1.55, 0, 0)
in disk_generate_states_env

Playing around with max_path_length (which would be the same as brownian_horizon),
the forcerange in xml file, and the density of the peg will allow for different behavior.
"""


def run_task(v):
    env = normalize(DiskGenerateStatesEnv())
    # env = normalize(Arm3dDiscEnv()) # "normal" disk environment

    # These two environments below test training a policy that moves the peg to a particular point
    # env = normalize(Arm3dMovePegEnv()) # tests moving peg
    # env = normalize(RobustDiskWrapperEnv(env = Arm3dDiscEnv()))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    if 'max_path_length' not in v:
        v['max_path_length'] = 300
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        plot=True,
        n_itr=400,
        max_path_length=v['max_path_length'],
        batch_size=10000,
    )
    algo.train()

run_task({})
vg = VariantGenerator()
vg.add('max_path_length', [300, 500])
for vv in vg.variants():
    run_experiment_lite(
        run_task,
        variant=vv,
        # Number of parallel workers for sampling
        n_parallel=4,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=2,
        exp_prefix="testpeg10",
        # variant=dict(),
        plot=True,
    )

