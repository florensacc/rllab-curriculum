"""
Run Atari in serial.
Separate conv nets for policy and baseline.
"""

# IMPORTS

# Serial:
# from rllab.sampler import parallel_sampler
# # from sandbox.adam.modified_sampler import parallel_sampler
# parallel_sampler.initialize(n_parallel=8)
# parallel_sampler.set_seed(1)

# from rllab.algos.trpo import TRPO
from sandbox.adam.dual_gpu.algos.trpo import TRPO
from rllab.policies.categorical_conv_policy import CategoricalConvPolicy
from rllab.baselines.gaussian_conv_baseline import GaussianConvBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

# Dual GPU:
# from sandbox.adam.dual_gpu.algos.dual_gpu_trpo import DualGpuTRPO
# from sandbox.adam.dual_gpu.categorical_conv_policy import CategoricalConvPolicy
# from sandbox.adam.dual_gpu.baselines.gaussian_conv_baseline import GaussianConvBaseline
# from sandbox.adam.gpar2.sampler.pairwise_gpu_sampler import PairwiseGpuSampler

# Common:
from sandbox.adam.dual_gpu.envs.atari import AtariEnv
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

###############################################################################
# SETUP

# Common:
# stub(globals())
env = AtariEnv(game="pong", obs_type="image", frame_skip=4)

policy_args = dict(
    name="policy",
    env_spec=env.spec,
    conv_filters=[16, 16],
    conv_filter_sizes=[4, 4],
    conv_strides=[2, 2],
    conv_pads=[(0, 0)] * 2,
    hidden_sizes=[20],
    eps=0.0,
)

baseline_optimizer = ConjugateGradientOptimizer(
    subsample_factor=0.2,
    num_slices=10,
)
baseline_regressor_args = dict(
    optimizer=baseline_optimizer,
    step_size=0.05,
    conv_filters=[16, 16],
    conv_filter_sizes=[4, 4],
    conv_strides=[2, 2],
    conv_pads=[(0, 0)] * 2,
    hidden_sizes=[20],
)
baseline_args = dict(
    env_spec=env.spec,
    regressor_args=baseline_regressor_args
)

# Serial:
policy = CategoricalConvPolicy(**policy_args)
policy_optimizer_args = dict(
    subsample_factor=0.2,
    num_slices=10,
)
baseline = GaussianConvBaseline(**baseline_args)
# baseline = ZeroBaseline(env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=10000,
    max_path_length=2000,  # env.horizon,
    n_itr=3,
    discount=0.99,
    step_size=0.05,
    optimizer_args=policy_optimizer_args,
    # sampler_cls=PairwiseGpuSampler,
    # sampler_args={'n_parallel': 7},
)

# Dual GPU:


###############################################################################
# RUN

algo.train()
