


from rllab.algos.npo import NPO
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.pchen.diag_npg.optimizers.diagonal_natural_gradient_optimizer import DiagonalNaturalGradientOptimizer

from sandbox.pchen.poga.algos.poga import POGA
stub(globals())

env = normalize(
    # CartpoleEnv()
SwimmerEnv()
)

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(32, 32)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)
# baseline = ZeroBaseline(env_spec=env.spec)

inner_algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=10000,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    step_size=0.03,
)
algo = POGA(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=50000,
    max_path_length=100,
    n_itr=50,
    discount=0.99,
    plot=True,
    inner_algo=inner_algo,
    optimizer=FirstOrderOptimizer(
        max_epochs=1,
        batch_size=64,
        learning_rate=1e-3,
    ),
)

run_experiment_lite(
    algo.train(),
    n_parallel=4,
    snapshot_mode="last",
    seed=1,
    mode="local",
    plot=True,
)
