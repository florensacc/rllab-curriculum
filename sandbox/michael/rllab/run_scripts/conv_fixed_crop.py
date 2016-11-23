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
    conv_filters=(8, 8),
    conv_filter_sizes=((5,5), (5,5)),
    conv_strides=((1, 1)),
    conv_pads=((0, 0)),
    hidden_sizes= (128,),
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
    exp_prefix="conv11",
    use_gpu=True
    # plot = True
)

# 4 decrease max path length to 3
#5 one layer of conv filters, simpler architecture
#6 fully connected net
#8 fully connected net, exact same architecture as MLP
#10 different architecture
#11 8 5x5 filters, does pretty well