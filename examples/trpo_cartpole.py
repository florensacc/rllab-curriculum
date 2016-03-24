from rllab.algos.trpo import TRPO
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

env = normalize(CartpoleEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)
algo = TRPO(
    batch_size=4000,
    whole_paths=True,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)

run_experiment_lite(
    algo.train(env=env, policy=policy, baseline=baseline),
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)
