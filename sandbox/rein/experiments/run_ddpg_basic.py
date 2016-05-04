import os
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy


from rllab.algos.ddpg import DDPG
os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import NormalizedEnv

from rllab.misc.instrument import stub, run_experiment_lite
import itertools

stub(globals())

# Param ranges
# seeds = range(10)
# mdp_classes = [CartpoleEnv, CartpoleSwingupEnv,
#                DoublePendulumEnv, MountainCarEnv]

seeds = [0]
mdp_classes = [DoublePendulumEnv]

mdps = [NormalizedEnv(env=mdp_class()) for mdp_class in mdp_classes]
param_cart_product = itertools.product(
    mdps, seeds
)

for mdp, seed in param_cart_product:

    policy = DeterministicMLPPolicy(env_spec=mdp.spec)

    baseline = GaussianMLPBaseline(
        mdp.spec,
        regressor_args=dict(hidden_sizes=(32,)),
    )

    qf = ContinuousMLPQFunction(env_spec=mdp.spec)
    es = OUStrategy(env_spec=mdp.spec)

    algo = DDPG(
        env=mdp,
        policy=policy,
        qf=qf,
        es=es,
        scale_reward=0.1,
        qf_learning_rate=0.001,
        policy_learning_rate=0.001,
        max_path_length=500,
        n_epochs=200,
        batch_size=4,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="ddpg-basic-d1",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        dry=False,
    )
