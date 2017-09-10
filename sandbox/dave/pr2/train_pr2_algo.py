import multiprocessing

from sandbox.dave.pr2.action_limiter import FixedActionLimiter
from sandbox.dave.rllab.algos.trpo import TRPO
from sandbox.dave.rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from rllab.envs.normalized_env import normalize
from sandbox.dave.rllab.goal_generators.pr2_goal_generators import PR2CrownGoalGeneratorSmall #PR2CrownGoalGeneratorSmall
from sandbox.dave.rllab.lego_generators.pr2_lego_generators import PR2LegoBoxBlockGeneratorSmall #PR2LegoBoxBlockGeneratorSmall #PR2LegoBoxBlockGeneratorSmall #PR2LegoFixedBlockGenerator
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.dave.rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from sandbox.dave.rllab.policies.gaussian_mlp_policy_tanh import GaussianMLPPolicy
from rllab.misc.instrument import VariantGenerator, variant


# from sandbox.dave.rllab.envs.mujoco.pr2_env_lego import Pr2EnvLego
from sandbox.dave.rllab.envs.mujoco.pr2_env_lego_position import Pr2EnvLego
from sandbox.dave.rllab.envs.mujoco.pr2_env_lego_hand import Pr2EnvLego
# from sandbox.dave.rllab.envs.mujoco.pr2_env_reach import Pr2EnvLego
# from sandbox.dave.rllab.envs.mujoco.pr2_env_lego_position_different_objects import Pr2EnvLego
from rllab import config
import os


# stub(globals())

train_goal_generator = PR2CrownGoalGeneratorSmall()
action_limiter = FixedActionLimiter()

# seeds = [1, 33]
seeds = [1]



env = normalize(Pr2EnvLego(
    goal_generator=train_goal_generator,
    lego_generator=PR2LegoBoxBlockGeneratorSmall(),
    # action_limiter=action_limiter,
    max_action=1,
    pos_normal_sample=True,
    qvel_init_std=0.01,
    pos_normal_sample_std=.01,  #0.5
    # use_depth=True,
    # use_vision=True,
    allow_random_restarts=True,
    # tip=t,
    # tau=t,
    # crop=True,
    # allow_random_vel_restarts=True,
))

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have n hidden layers, each with k hidden units.
    hidden_sizes=(64, 64, 64),
    init_std=0.1,
    output_gain=0.1,
    # pkl_path= "data/local/train-Lego/IROS/3Dangle/exp_2/params.pkl"
    # json_path="/home/young_clgan/GitRepos/rllab-goals/data/local/train-Lego/rand_init_angle_reward_shaping_continuex2_2016_10_17_12_48_20_0001/params.json",
    )

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=50000,
    max_path_length=150,  #100
    n_itr=15000, #50000
    discount=0.95,
    gae_lambda=0.98,
    step_size=0.01,
    goal_generator=train_goal_generator,
    action_limiter=None,
    optimizer_args={'subsample_factor': 0.1},
    # discount_weights={'angle': 0.1, 'tip': .1},
    # plot=True, # breaks if plot trues
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    )

algo.train()
