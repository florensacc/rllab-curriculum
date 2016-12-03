from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_conv_policy import GaussianConvPolicy
from sandbox.michael.fixed_crop_env import FixedCropEnv

# from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline

stub(globals())

env = normalize(FixedCropEnv(max_action= 6, radius = 10, max_init_distance= 6))
policy = GaussianConvPolicy(
    hidden_sizes= (128,128),
    env_spec=env.spec,
)
baseline = GaussianMLPBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    max_path_length=3,
    n_itr=10000, # iterations
    batch_size= 8000, # trajectories = batch size / max_path_length
    baseline=baseline,
    # plot = True
)
# algo.train()
run_experiment_lite(
    algo.train(),
    n_parallel=4,
    snapshot_mode="last",
    exp_prefix="conv4",
    use_gpu=True
    # plot = True
)

# 4 decrease max path length to 3