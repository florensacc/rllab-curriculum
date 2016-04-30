import os
from sandbox.rein.algos.ddpg_vbnn import DDPG
os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.rein.algos.trpo_unn import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
import itertools

stub(globals())

# Param ranges
# seeds = range(10)
# etas = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
# normalize_rewards = [False, True]
# kl_ratios = [False, True]
# mdp_classes = [CartpoleSwingupEnv,
#                DoublePendulumEnv, MountainCarEnv]

seeds = range(1)
etas = [0.01]
normalize_rewards = [False]
kl_ratios = [False]
mdp_classes = [MountainCarEnv]

mdps = [NormalizedEnv(env=mdp_class())
        for mdp_class in mdp_classes]
param_cart_product = itertools.product(
    kl_ratios, normalize_rewards, mdps, etas, seeds
)

for kl_ratio, normalize_reward, mdp, eta, seed in param_cart_product:

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
        n_epochs=1000,
        eta=eta,
        eta_discount=1.0,
        snn_n_samples=10,
        use_reverse_kl_reg=True,
        use_replay_pool=True,
        use_kl_ratio=kl_ratio,
        use_kl_ratio_q=kl_ratio,
        n_itr_update=1,
        kl_batch_size=5,
        normalize_reward=normalize_reward,
        stochastic_output=False,
        dyn_replay_pool_size=100000,
        n_updates_per_sample=500,
        second_order_update=False,
        unn_n_hidden=[32],
        unn_layers_type=[1, 1],
        unn_learning_rate=0.0001,
        dyn_replay_freq=100
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="ddpg-expl-basic-a1",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        dry=False,
        script="scripts/run_experiment_lite.py",
    )
