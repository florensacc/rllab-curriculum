from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv
# from rllab.envs.mujoco.hopper_env import HopperEnv  # if Gym don't import this!!
import gym

import datetime
import dateutil.tz
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

stub(globals())

env_name='Walker2d-v1'
env = normalize(GymEnv(env_name))
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
    batch_size=10000,
    max_path_length=500,
    n_itr=400,
    discount=0.99,
    step_size=0.01,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

for seed in [2,4,19,33]:
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    run_experiment_lite(  #the scripts/run_experiment_lite.py takes care of initializing the logger and passing args!
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        # Set where to save things
        exp_name="{}_gauss_N{}_T{}_n_iter{}_s{}_{}".format(
            env_name, 10000, 500, 400, seed, timestamp),
            # algo.batch_size, algo.max_path_length, algo.n_itr, seed),
        exp_prefix=env_name,
        plot=False,
    )
