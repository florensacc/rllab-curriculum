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
etas = [0, 0.0001, 0.001, 0.01, 0.1]
normalize_rewards = [False]
kl_ratios = [False]
update_likelihood_sds = [False, True]
# mdp_classes = [CartpoleEnv, CartpoleSwingupEnv, DoublePendulumEnv,
#                MountainCarEnv, DoublePendulumEnvX, MountainCarEnvX,
#                CartpoleSwingupEnvX]
mdp_classes = [CartpoleSwingupEnv, CartpoleSwingupEnvX]
mdps = [mdp_class()
        for mdp_class in mdp_classes]

param_cart_product = itertools.product(
    update_likelihood_sds, kl_ratios, normalize_rewards, mdps, etas, seeds
)

for update_likelihood_sd, kl_ratio, normalize_reward, mdp, eta, seed in param_cart_product:

    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(32,),
    )

    baseline = GaussianMLPBaseline(
        mdp.spec,
        regressor_args=dict(hidden_sizes=(32,),
                            batchsize=1000),
    )

    algo = TRPO(
        # TRPO settings
        # -------------
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=5000,
        whole_paths=True,
        max_path_length=500,
        n_itr=1000,
        step_size=0.01,
        optimizer_args=dict(num_slices=2),
        # -------------

        # VIME settings
        # -------------
        eta=eta,
        snn_n_samples=10,
        use_replay_pool=False,
        use_kl_ratio=kl_ratio,
        use_kl_ratio_q=kl_ratio,
        kl_batch_size=4,
        normalize_reward=normalize_reward,
        replay_pool_size=100000,
        n_updates_per_sample=10000,
        second_order_update=False,
        state_dim=mdp.spec.observation_space.shape,
        action_dim=(mdp.spec.action_space.flat_dim,),
        reward_dim=(1,),
        layers_disc=[
            dict(name='gaussian',
                 n_units=32),
            dict(name='outerprod'),
            dict(name='gaussian',
                 n_units=32),
            dict(name='split',
                 n_units=32),
            dict(name='gaussian',
                 n_units=mdp.spec.observation_space.shape[0],
                 nonlinearity=lasagne.nonlinearities.linear),
        ],
        unn_learning_rate=0.005,
        surprise_transform=None,  # BatchPolopt.SurpriseTransform.CAP90PERC,
        update_likelihood_sd=update_likelihood_sd,
        replay_kl_schedule=0.98,
        output_type=BNN.OutputType.REGRESSION,
        pool_batch_size=32,
        likelihood_sd_init=0.1,
        prior_sd=0.05,
        # -------------
        disable_variance=False,
        group_variance_by=BNN.GroupVarianceBy.WEIGHT,
        surprise_type=BNN.SurpriseType.COMPR,
        predict_reward=True,
        use_local_reparametrization_trick=True,
        n_itr_update=1,
        # -------------
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-vime-basic-a",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="lab_kube",
        dry=False,
        use_gpu=False,
        script="sandbox/rein/experiments/run_experiment_lite.py",
    )
