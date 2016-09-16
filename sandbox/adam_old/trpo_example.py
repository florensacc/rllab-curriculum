# from rllab.algos.trpo import TRPO
from sandbox.adam.trpo_timed import TRPO_timed
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from sandbox.adam.trpo_par import TRPO_par
# from sandbox.adam.linear_feature_baseline_par import LinearFeatureBaseline_par

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy



env = normalize(CartpoleEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO_timed(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=2000,
    max_path_length=100,
    n_itr=4,
    discount=0.99,
    step_size=0.01,
    n_proc=2,
)
algo.train()