# from examples.distract_crop_env import DistractCropEnv
from sandbox.michael.variable_crop_env import VariableCropEnv
from rllab.policies.gaussian_conv_policy import GaussianConvPolicy

from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.two_gaussian_conv_policy import TwoGaussianConvPolicy

stub(globals())

env = normalize(VariableCropEnv(max_action= 6, radius = 10, max_init_distance= 6))
policy = GaussianConvPolicy(
    conv_filters=(8, 8),
    conv_filter_sizes=((5,5), (5,5)),
    conv_strides=((1, 1)),
    conv_pads=((0, 0)),
    hidden_sizes= (256,),
    env_spec=env.spec,
)
baseline = GaussianMLPBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    max_path_length=2,
    n_itr=8000, # iterations
    batch_size= 1000, # trajectories = batch size / max_path_length
    baseline=baseline,
    optimizer_args={'num_slices': 100},
    # plot = True
)
# algo.train()
run_experiment_lite(
    algo.train(),
    n_parallel=1,
    snapshot_mode="last",
    exp_prefix="convvariable10",
    use_gpu=True
    # plot = True
)

#2 done = dist_x <= self.correctness and dist_y <= self.correctness and scale_diff < 0.1
#  reward = - (dist_x ** 2 + dist_y ** 2) ** 0.5 - scale_diff * 2
# 3 switch to 256 instead of 128, correctness condition
# 5 add noise to environment

# env = normalize(DistractCropEnv(max_action= 6, radius = 10, max_init_distance= 6, max_scale_action = 1.2))
# policy = TwoGaussianConvPolicy(
#     conv_filters=(16, 16),
#     conv_filter_sizes=((5,5), (5,5)),
#     conv_strides=((1, 1)),
#     conv_pads=((0, 0)),
#     hidden_sizes= (256,),
#     env_spec=env.spec,
# )
# baseline = GaussianMLPBaseline(env_spec=env.spec)
# algo = TRPO(
#     env=env,
#     policy=policy,
#     max_path_length=3,
#     n_itr=10000, # iterations
#     batch_size= 10000, # trajectories = batch size / max_path_length
#     baseline=baseline,
#     # plot = True
# )
# # algo.train()
# run_experiment_lite(
#     algo.train(),
#     # n_parallel=4,
#     snapshot_mode="last",
#     exp_prefix="distractor",
#     exp_name="aspectscale3",
#     use_gpu=True
#     # plot = True
# )

# distractor 2 works well, trained for always going after blue circle (target == 1)
# distractor/aspect scale 1, allow both sides to scale independently, more movement
# distractor/aspect scale 2, allow both sides to scale independently, max_scale_action = 1.2, path length = 3
# distractor/aspect scale 3, allow both sides to scale independently, more conv filters