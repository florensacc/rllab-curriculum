
from sandbox.adam.forrest.empgo import EMPGO
from rllab.algos.vpg import VPG
from sandbox.adam.forrest.emtrpo import EMTRPO
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
# from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.adam.forrest.options_mlp_policy import OptionsMLPPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

stub(globals())

# env = normalize(MountainCarEnv())
# env = normalize(SwimmerEnv())
env = normalize(GymEnv("BipedalWalker-v2", record_video=False))
# env = normalize(GymEnv("MountainCarContinuous-v0"))

baseline = LinearFeatureBaseline(env_spec=env.spec)

# policy = GaussianMLPPolicy(
#     env_spec=env.spec,
#     hidden_sizes=(32,32),
# )
#
# algo = TRPO(
#     env=env,
#     policy=policy,
#     baseline=baseline,
#     batch_size=10000,
#     max_path_length=1000,
#     n_itr=1000,
#     discount=0.99,
#     step_size=0.01
# )

policy = OptionsMLPPolicy(
    env_spec=env.spec,
    num_options=4,
    hidden_sizes=((32, 32), (32, 32), (32, 32))
)

algo = EMTRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=10000,
    max_path_length=500,
    n_itr=10000,
    discount=0.99,
    step_size=0.01,
)

run_experiment_lite(
    algo.train(),
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    exp_prefix='bipedal-options-longhaul'
)
