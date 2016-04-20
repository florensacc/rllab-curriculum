import os
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
import itertools

stub(globals())

# Param ranges
seeds = range(10)
etas = [0.0001, 0.0003, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
mdp_classes = [CartpoleEnv, CartpoleSwingupEnv,
               DoublePendulumEnv, MountainCarEnv]
mdps = [NormalizedEnv(env=mdp_class())
        for mdp_class in mdp_classes]
param_cart_product = itertools.product(
    mdps, etas, seeds
)

for mdp, eta, seed in param_cart_product:

    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(32,),
    )

    baseline = GaussianMLPBaseline(
        mdp.spec,
        regressor_args=dict(hidden_sizes=(32,)),
    )

    batch_size = 5000
    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=500,
        n_itr=1000,
        step_size=0.01,
        eta=eta,
        eta_discount=1.0,
        snn_n_samples=10,
        subsample_factor=1.0,
        use_reverse_kl_reg=True,
        use_replay_pool=True,
        use_kl_ratio=True,
        use_kl_ratio_q=True,
        n_itr_update=5,
        kl_batch_size=5,
        normalize_reward=False,
        stochastic_output=False,
        replay_pool_size=100000,
        n_updates_per_sample=500,
        #         second_order_update=True,
        unn_n_hidden=[32],
        unn_layers_type=[1, 1],
        unn_learning_rate=0.001
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-expl-basic-v2x",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="lab_kube",
        dry=False,
    )
