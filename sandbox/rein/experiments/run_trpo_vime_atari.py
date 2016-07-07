import os
from sandbox.rein.envs.mountain_car_env_x import MountainCarEnvX
from sandbox.rein.envs.double_pendulum_env_x import DoublePendulumEnvX
from rllab.envs.gym_env import GymEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.rein.algos.trpo_vime import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools
os.environ["THEANO_FLAGS"] = "device=gpu"

stub(globals())

# Param ranges
# seeds = range(10)
# etas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
# normalize_rewards = [False]
# kl_ratios = [True]
# mdp_classes = [MountainCarEnv]
# mdps = [NormalizedEnv(env=mdp_class())
#         for mdp_class in mdp_classes]

seeds = range(3)
etas = [0, 0.000001]
normalize_rewards = [False]
kl_ratios = [True]
mdps = [GymEnv("Freeway-ram-v0")]
# mdp_classes = [MountainCarEnvX]

param_cart_product = itertools.product(
    kl_ratios, normalize_rewards, mdps, etas, seeds
)

for kl_ratio, normalize_reward, mdp, eta, seed in param_cart_product:

    policy = CategoricalMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(64, 64)
    )

    baseline = GaussianMLPBaseline(
        mdp.spec,
        regressor_args=dict(hidden_sizes=(64, 64)),
    )

    algo = TRPO(
        # TRPO settings
        # -------------
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=50000,
        whole_paths=True,
        max_path_length=5000,
        n_itr=250,
        step_size=0.01,
        subsample_factor=1.0,
        # -------------

        # VIME settings
        # -------------
        eta=eta,
        snn_n_samples=20,
        use_replay_pool=True,
        use_kl_ratio=kl_ratio,
        use_kl_ratio_q=kl_ratio,
        kl_batch_size=128,
        normalize_reward=normalize_reward,
        replay_pool_size=1000000,
        n_updates_per_sample=50000,
        second_order_update=True,
        unn_n_hidden=[512, 512],
        unn_layers_type=['gaussian', 'gaussian', 'gaussian'],
        unn_learning_rate=0.001,
        surprise_transform='log(1+surprise)',
        update_likelihood_sd=False,
        replay_kl_schedule=0.99,
        output_type='regression',
        pool_batch_size=128,
        likelihood_sd_init=0.1,
        prior_sd=0.5,
        # -------------
        disable_variance=False,
        group_variance_by='weight',
        surprise_type='information_gain',
        predict_reward=False,
        use_local_reparametrization_trick=True,
        n_itr_update=1,
        # -------------
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-vime-freeway-e",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="lab_kube",
        dry=False,
        use_gpu=True,
        script="sandbox/rein/experiments/run_experiment_lite.py",
    )
