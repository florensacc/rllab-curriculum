from sandbox.michael.single_video_env import SingleVideoEnv

from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.two_gaussian_conv_policy import TwoGaussianConvPolicy

stub(globals())

env = normalize(SingleVideoEnv(image_folder = "01-Light/",
        image_name = "01-Light_video00003/",
        annotation_file_name = "01-Light_video00003.ann",
                               side = 50, min_scale=0.66, max_scale=1.5, fixed_sample=False))

policy = TwoGaussianConvPolicy(
    conv_filters=(32, 32),
    conv_filter_sizes=((5,5), (5,5)),
    conv_strides=((1, 1)),
    conv_pads=((0, 0)),
    hidden_sizes= (128,128),
    output_nonlinearity=None,
    env_spec=env.spec,
)

baseline = GaussianMLPBaseline(env_spec=env.spec)
# baseline = GaussianConvBaseline(env_spec=env.spec,
#                                 conv_filters=(16, 16),
#                                 conv_filter_sizes=((5,5), (5,5)),
#                                 conv_strides=((1, 1)),
#                                 conv_pads=((0, 0)),
#                                 hidden_sizes= (64,64),) #TODO: figure out
# baseline = ZeroBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    max_path_length=2,
    n_itr=20000, # iterations
    batch_size= 8000, # trajectories = batch size / max_path_length
    baseline=baseline,
    optimizer_args={'num_slices': 20},
    # plot = True
)
# algo.train()
run_experiment_lite(
    algo.train(),
    n_parallel=1,
    snapshot_mode="last",
    exp_prefix="video",
    exp_name="practice22",
    use_gpu=True
    # plot = True
)
# -290.333
#up to practice5 (all the same image, made actions bigger to 50px)
#6 optimizer_args={'num_slices': 10},

#7 change reward function
#8 negative reward function
#9 max path length to 3
#10 set up actions to 0.6 and 1.5
#12 increase number of filters
#16 switch to 1.5 times original size
#18  max path length 1, trained to completion (single video worked)
#19 output nonlinearty none (does not seem to have effect
#21 change self.max_action to 2
