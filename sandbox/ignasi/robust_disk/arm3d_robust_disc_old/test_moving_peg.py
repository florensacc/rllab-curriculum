from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.point_env import PointEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite, VariantGenerator
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.ignasi.envs.action_limited_env import ActionLimitedEnv
from sandbox.ignasi.envs.arm3d.arm3d_disc_env import Arm3dDiscEnv
from sandbox.ignasi.envs.arm3d.arm3d_move_peg_env import Arm3dMovePegEnv
from sandbox.ignasi.envs.arm3d.arm3d_wrapper_env import RobustDiskWrapperEnv


def run_task(v):
    # env = normalize(Arm3dMovePegEnv()) # tests moving peg
    env = normalize(RobustDiskWrapperEnv(env = Arm3dDiscEnv()))

    policy = GaussianMLPPolicy(
        env_spec=env.spec, # make sure 2 actions
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    if 'max_path_length' not in v:
        v['max_path_length'] = 500
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        plot=True,
        n_itr=400,
        max_path_length=v['max_path_length'],
        batch_size=30000,
    )
    algo.train()

# run_task()
# try different max_path_length?

# TODO: visualize target
# TODO: final distance is what matters

run_task({})
vg = VariantGenerator()
vg.add('max_path_length', [200])
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

# testpeg6 time step 0.02
# testpeg7 time step 0.01
#testpeg9 time step 0.02