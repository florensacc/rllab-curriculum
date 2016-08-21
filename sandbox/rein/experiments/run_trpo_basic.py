import os
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rein.dynamics_models.bnn.bnn import BNN
from rllab.envs.normalized_env import NormalizedEnv
from rllab.envs.box2d.cartpole_env import CartpoleEnv

import lasagne

from sandbox.rein.algos.trpo_vime import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools
from sandbox.rein.algos.batch_polopt_vime import BatchPolopt
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from sandbox.rein.envs.double_pendulum_env_x import DoublePendulumEnvX
from sandbox.rein.envs.mountain_car_env_x import MountainCarEnvX
from sandbox.rein.envs.cartpole_swingup_env_x import CartpoleSwingupEnvX
from rllab.envs.box2d.mountain_car_env import MountainCarEnv

os.environ["THEANO_FLAGS"] = "device=cpu"

stub(globals())

# Param ranges
seeds = range(10)
etas = [0, 0.001]
# seeds = [0]
# etas = [0.1]
batch_sizes = [5000]
normalize_rewards = [False]
kl_ratios = [False]
update_likelihood_sds = [True]
mdp_classes = [CartpoleSwingupEnvX]
mdps = [mdp_class() for mdp_class in mdp_classes]

param_cart_product = itertools.product(
    batch_sizes, update_likelihood_sds, kl_ratios, normalize_rewards, mdps, etas, seeds
)

for batch_size, update_likelihood_sd, kl_ratio, normalize_reward, mdp, eta, seed in param_cart_product:
    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(32,),
    )

    baseline = GaussianMLPBaseline(
        mdp.spec,
        regressor_args=dict(hidden_sizes=(32,),
                            batchsize=1000000),
    )

    algo = TRPO(
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=1000,
        whole_paths=True,
        max_path_length=500,
        n_itr=250,
        step_size=0.01,
        optimizer_args=dict(
            num_slices=1,
            subsample_factor=0.1),
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-vime-basic-a",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        dry=False,
        use_gpu=False,
        script="sandbox/rein/experiments/run_experiment_lite.py",
    )
