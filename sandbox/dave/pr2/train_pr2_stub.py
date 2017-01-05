import multiprocessing

from sandbox.dave.pr2.action_limiter import FixedActionLimiter
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.dave.rllab.envs.mujoco.pr2_env_lego import Pr2EnvLego
from rllab.envs.normalized_env import normalize
from sandbox.dave.rllab.goal_generators.pr2_goal_generators import PR2CrownGoalGeneratorSmall
from sandbox.dave.rllab.lego_generators.pr2_lego_generators import PR2LegoBoxBlockGeneratorSmall
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.dave.rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import numpy as np
stub(globals())

train_goal_generator = PR2CrownGoalGeneratorSmall()
action_limiter = FixedActionLimiter()

env = normalize(Pr2EnvLego(
    goal_generator=train_goal_generator,
    lego_generator=PR2LegoBoxBlockGeneratorSmall(),
    action_limiter=action_limiter,
    max_action=1,
    pos_normal_sample=True,
    qvel_init_std=0.01,
    pos_normal_sample_std=1,
    # use_depth=True,
    use_vision=False,
    allow_random_restarts=True,
    allow_random_vel_restarts=True,
    ))

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have n hidden layers, each with k hidden units.
    hidden_sizes=(64, 64, 64),
    # output_gain=1,
    init_std=1,
    pkl_path="/home/ignasi/GitRepos/rllab-private/data/s3/train-Lego/state/random_random_pixel_penalty_p0005_d_06_reward_distance_1_angle_02_crown_normal_sample_001_50000/params.pkl"
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
    optimizer_args={'subsample_factor': 0.1}
    # plot=True,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    )

run_experiment_lite(
    algo.train(),
    use_gpu=False,
    # Number of parallel workers for sampling
    n_parallel=8,
    # n_parallel=4,
    sync_s3_pkl=True,
    periodic_sync=True,
    # sync_s3_png=True,
    aws_config={"spot_price": '3.5', 'instance_type': 'm4.4xlarge'},
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # python_command="apt-get update \
    # && apt-get -y install \
    # xorg-dev \
    # libglu1-mesa libgl1-mesa-dev \
    # xvfb \
    # libxinerama1 libxcursor1 \
    # xvfb-run -a -s \"-screen 0 1400x900x24 +extension RANDR\" -- glxinfo",
    # && xvfb-run -a \"-screen 0 1400x900x24 +extension RANDR\" -- python",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # mode="local",
    mode="ec2",
    # log_dir="data/local/train-Lego/trial_pretraining",
    exp_prefix="train-Lego/state",
    exp_name="random_random_pixel_penalty_p0005_d_06_reward_distance_1_angle_02_crown_normal_sample_1_50000_small_lego",
    # plot=True,
)
