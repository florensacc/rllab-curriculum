import multiprocessing

from sandbox.dave.pr2.action_limiter import FixedActionLimiter
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.dave.rllab.envs.mujoco.pr2_env_lego import Pr2EnvLego
from rllab.envs.normalized_env import normalize
from sandbox.dave.rllab.goal_generators.pr2_goal_generators import PR2BoxGoalGenerator
from sandbox.dave.rllab.lego_generators.pr2_lego_generators import PR2LegoFixedBlockGenerator
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.dave.rllab.policies.pretrain_gaussian_mlp_policy import PretrainGaussianMLPPolicy

stub(globals())

train_goal_generator = PR2BoxGoalGenerator(small_range=True)
action_limiter = FixedActionLimiter()

env = normalize(Pr2EnvLego(
    goal_generator=train_goal_generator,
    lego_generator=PR2LegoFixedBlockGenerator(),
    action_limiter=action_limiter,
    max_action=1,
    pos_normal_sample=True,
    qvel_init_std=0.01
    ))

policy = PretrainGaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have n hidden layers, each with k hidden units.
    hidden_sizes=(64, 64, 64),
    # output_gain=1,
    init_std=1,
    npz_path="/home/young_clgan/GitRepos/rllab-private/sandbox/dave/data/params/finetune/train139/params.npz",
    json_path="/home/young_clgan/GitRepos/rllab-private/sandbox/dave/data/params/finetune/train139/params.json",
    )

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=50000,
    max_path_length=100,  #100
    n_itr=500000, #50000
    discount=0.95,
    gae_lambda=0.98,
    step_size=0.01,
    goal_generator=train_goal_generator,
    action_limiter=action_limiter,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    plot=True,
)

run_experiment_lite(
    algo.train(),
    use_gpu=True,
    # Number of parallel workers for sampling
    n_parallel=multiprocessing.cpu_count(),
    # n_parallel=2,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # log_dir="data/local/train-Lego/trial_pretraining",
    exp_prefix="train-Lego/fixed_pretraining",
    exp_name="fixed_pretraining",
    plot=True,
)
