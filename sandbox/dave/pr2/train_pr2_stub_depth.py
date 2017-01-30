import multiprocessing
from sandbox.dave.sampler.pairwise_gpu_sampler import PairwiseGpuSampler
from sandbox.dave.pr2.action_limiter import FixedActionLimiter
from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from sandbox.dave.rllab.baselines.conv_mlp_baseline import GaussianConvMLPBaseline
from sandbox.dave.rllab.envs.mujoco.pr2_env_lego import Pr2EnvLego
from rllab.envs.normalized_env import normalize
from sandbox.dave.rllab.goal_generators.pr2_goal_generators import PR2FixedGoalGenerator
from sandbox.dave.rllab.lego_generators.pr2_lego_generators import PR2LegoFixedBlockGenerator
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.dave.rllab.policies.depth_gaussian_mlp_policy import DepthGaussianMLPPolicy
# from sandbox.dave.rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import cProfile
import re


stub(globals())

train_goal_generator = PR2FixedGoalGenerator()
action_limiter = FixedActionLimiter()

env = normalize(Pr2EnvLego(
    goal_generator=train_goal_generator,
    lego_generator=PR2LegoFixedBlockGenerator(),
    action_limiter=action_limiter,
    max_action=1,
    pos_normal_sample=True,
    qvel_init_std=0.01,
    use_depth=True,
    use_vision=True,
    ))

policy = DepthGaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have n hidden layers, each with k hidden units.
    hidden_sizes=(64, 64, 64),
    # output_gain=1,
    init_std=1,
    npz_path="/home/ignasi/GitRepos/rllab-private/data/local/pkl_files/caffe_AlexNet/python3/params.npz",
    # json_path="/home/ignasi/GitRepos/rllab-goals/data/local/train-Lego/rand_init_angle_reward_shaping_continuex2_2016_10_17_12_48_20_0001/params.json",
    )

baseline = GaussianConvMLPBaseline(env_spec=env.spec,
                                # subsample_factor=0.2,
                                regressor_args={'subsample_factor': 0.1,
                                                'npz_path':"/home/ignasi/GitRepos/rllab-private/data/local/pkl_files/caffe_AlexNet/python3/params.npz"},
                                optimizer_args={'num_slices': 500,},
                                )
# baseline = ZeroBaseline(env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=5000,
    max_path_length=10,  #100
    n_itr=50000,
    discount=0.95,
    gae_lambda=0.98,
    step_size=0.01,
    goal_generator=train_goal_generator,
    action_limiter=action_limiter,
    optimizer_args={'num_slices': 500, 'subsample_factor': 0.1},
    # sampler_cls=PairwiseGpuSampler,
    # sampler_args={'n_parallel': 5}
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

run_experiment_lite(
    algo.train(),
    use_gpu=True,
    # Number of parallel workers for sampling
    # n_parallel=multiprocessing.cpu_count(),
    # n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    # python_command="python",
    seed=1,
    # log_dir="data/local/train-Lego/trial_pretraining",
    exp_prefix="train-Lego/RSS",
    exp_name="everything_fixed_depth",
    # plot=True,
)
