from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv
# from rllab.envs.mujoco.hopper_env import HopperEnv  # if Gym don't import this!!

stub(globals())

env = normalize(GymEnv('Hopper-v0'))
# env = HopperEnv()

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
    batch_size=  500,
    max_path_length=100,
    n_itr=4,
    discount=0.99,
    step_size=0.01,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

run_experiment_lite(  #the scripts/run_experiment_lite.py takes care of initializing the logger and passing args!
    algo.train(),
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # Set where to save things
    exp_name="hopper_snn_try",
    exp_prefix="hopeer_snn",
    plot=True,
)
