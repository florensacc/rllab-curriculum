"""
Run Atari with parallel optimization using synkhronos.
Separate conv nets for policy and baseline.
"""


# IMPORTS
# # from sandbox.adam.modified_sampler import parallel_sampler

import os
os.environ['MKL_NUM_THREADS'] = "1"
from rllab.sampler import parallel_sampler
parallel_sampler.initialize(n_parallel=8)
parallel_sampler.set_seed(1)

import synkhronos
synkhronos.fork()  # before building any theano variables, graphs, functions
from sandbox.adam.synkhronos.algos.trpo import TRPO

# from rllab.algos.trpo import TRPO


from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

# Dual GPU:
# from sandbox.adam.dual_gpu.algos.dual_gpu_trpo import DualGpuTRPO
# from sandbox.adam.dual_gpu.categorical_conv_policy import CategoricalConvPolicy
# from sandbox.adam.dual_gpu.baselines.gaussian_conv_baseline import GaussianConvBaseline
# from sandbox.adam.gpar2.sampler.pairwise_gpu_sampler import PairwiseGpuSampler


# from sandbox.adam.gpar2.envs.humanoid_mine import HumanoidEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize

###############################################################################
# SETUP

# Common:
# stub(globals())
env = normalize(CartpoleEnv())
# env.step = env.step_two_frame_test

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=5000,
    max_path_length=500,  # env.horizon,
    n_itr=40,
    discount=0.99,
    step_size=0.05,
    # sampler_cls=PairwiseGpuSampler,
    # sampler_args={'n_parallel': 7},
)

# Dual GPU:


###############################################################################
# RUN

algo.train()
