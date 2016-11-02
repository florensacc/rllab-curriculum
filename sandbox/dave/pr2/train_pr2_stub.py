import multiprocessing

from sandbox.dave.pr2.action_limiter import FixedActionLimiter
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.dave.rllab.envs.mujoco.pr2_env_lego import Pr2EnvLego
from rllab.envs.normalized_env import normalize
from sandbox.dave.rllab.goal_generators.pr2_goal_generators import PR2BoxGoalGeneratorSmall
from sandbox.dave.rllab.lego_generators.pr2_lego_generators import PR2LegoBoxBlockGeneratorSmall
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.dave.rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

stub(globals())

train_goal_generator = PR2BoxGoalGeneratorSmall()
action_limiter = FixedActionLimiter()

env = normalize(Pr2EnvLego(
    goal_generator=train_goal_generator,
    lego_generator=PR2LegoBoxBlockGeneratorSmall(),
    action_limiter=action_limiter,
    max_action=1,
    pos_normal_sample=False,
    qvel_init_std=0.01,
    use_depth=True,
    use_vision=True,
    ))

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have n hidden layers, each with k hidden units.
    hidden_sizes=(64, 64, 64),
    # output_gain=1,
    init_std=1,
    # pkl_path="/home/ignasi/GitRepos/rllab-private/data/local/pkl_files/caffe_AlexNet/caffe_reference.pkl",
    # json_path="/home/ignasi/GitRepos/rllab-goals/data/local/train-Lego/rand_init_angle_reward_shaping_continuex2_2016_10_17_12_48_20_0001/params.json",
    )

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=50000,
    max_path_length=150,  #100
    n_itr=500000, #50000
    discount=0.95,
    gae_lambda=0.98,
    step_size=0.01,
    goal_generator=train_goal_generator,
    action_limiter=action_limiter,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

run_experiment_lite(
    algo.train(),
    # use_gpu=True,
    # Number of parallel workers for sampling
    # n_parallel=multiprocessing.cpu_count(),
    n_parallel=32,
    sync_s3_pkl=True,
    # sync_s3_png=True,
    aws_config={"spot_price": '1.5', 'instance_type': 'm4.16xlarge'},
    # Only keep the snapshot parameters for the last iteration
    # snapshot_mode="all",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    mode="ec2",
    # log_dir="data/local/train-Lego/trial_pretraining",
    exp_prefix="train-Lego/state",
    exp_name="random_goals_random_lego",
    # plot=True,
)
