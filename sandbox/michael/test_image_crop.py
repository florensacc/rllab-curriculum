from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from examples.image_crop_env import ImageCropEnv
from examples.fixed_crop_env import FixedCropEnv

stub(globals())

# Tests the fixed crop environment
env = normalize(FixedCropEnv(max_action= 6, radius = 10, max_init_distance= 6))
policy = GaussianMLPPolicy(
    hidden_sizes= (256, 256),
    env_spec=env.spec,
)
baseline = GaussianMLPBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    max_path_length=3,
    n_itr=10000, # iterations
    batch_size= 10000, # trajectories = batch size / max_path_length
    baseline=baseline,
    # plot = True
)
# algo.train()
run_experiment_lite(
    algo.train(),
    n_parallel=4,
    snapshot_mode="last",
    exp_prefix="fixedcrop10",
    use_gpu=True
    # plot = True
)

