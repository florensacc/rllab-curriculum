"""
Run Atari on dual-gpu machine.
Separate conv nets for policy and baseline.
Baseline is fit on one GPU while the sampling is performed with the other.
Both GPUs used to optimize policy.
"""

from sandbox.adam.dual_gpu.algos.dual_gpu_trpo import DualGpuTRPO
from sandbox.adam.dual_gpu.baselines.gaussian_conv_baseline import GaussianConvBaseline

# from rllab.envs.gym_env import GymEnv
from sandbox.adam.gpar2.envs.humanoid_mine import HumanoidEnv
# from sandbox.adam.gpar.envs.humanoid_baseline import HumanoidEnv

# from rllab.envs.mujoco.humanoid_env import HumanoidEnv
# from rllab.envs.normalized_env import normalize
from sandbox.adam.gpar2.envs.normalized_env_mine import normalize
# from rllab.misc.instrument import stub, run_experiment_lite
# from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.adam.gpar2.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from sandbox.adam.gpar.policies.gaussian_mlp_policy_notran import GaussianMLPPolicy

from sandbox.adam.gpar2.sampler.pairwise_gpu_sampler import PairwiseGpuSampler
# from sandbox.adam.gpar.sampler.pairwise_gpu_multi_sampler import PairwiseGpuMultiSampler

from timeit import default_timer as timer
# stub(globals())

# env = normalize(GymEnv("Humanoid-v1", record_video=False, record_log=False))
env = normalize(HumanoidEnv())
# env = normalize(GymEnv("Hopper-v1", record_video=False, record_log=False))

# policy = GaussianMLPPolicy(
#     env_spec=env.spec,
#     hidden_sizes=(300, 300, 300)
# )

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = MultiGpuTRPO(
    env=env,
    # policy=policy,
    policy_cls=GaussianMLPPolicy,
    policy_args=dict(
        env_spec=env.spec,
        hidden_sizes=(300, 300, 300),
    ),
    baseline=baseline,
    batch_size=6000,
    max_path_length=200,  # env.horizon,
    n_itr=5,
    discount=0.99,
    step_size=0.01,
    n_gpu=2,
    sampler_cls=PairwiseGpuSampler,
    # sampler_cls=PairwiseGpuMultiSampler,
    sampler_args={'n_parallel': 7,
                  # 'n_simulators': 10,
                  }
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

# run_experiment_lite(
#     algo.train(timer_name='ser_8_mkl1'),
#     # Different script initializes the modified parallel sampler
#     script="sandbox/adam/modified_sampler/run_experiment_lite.py",
#     # Number of parallel workers for sampling
#     n_parallel=4,
#     # Only keep the snapshot parameters for the last iteration
#     snapshot_mode="last",
#     # Specifies the seed for the experiment. If this is not provided, a random seed
#     # will be used
#     seed=1,
#     # plot=True,
#     exp_prefix='timing9',
#     use_gpu=False,
# )
t0 = timer()
algo.train()
t1 = timer()
print("Total training time: ", t1 - t0)
