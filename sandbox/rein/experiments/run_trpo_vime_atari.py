import os
from sandbox.rein.envs.mountain_car_env_x import MountainCarEnvX
from sandbox.rein.envs.double_pendulum_env_x import DoublePendulumEnvX
from rllab.envs.gym_env import GymEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rein.dynamics_models.bnn.bnn import BNN
from rllab.core.network import ConvNetwork

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rein.algos.trpo_vime import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools
from sandbox.rein.algos.batch_polopt_vime import BatchPolopt
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
seeds = [0, 1]
etas = [0.1]
# seeds = range(5)
# etas = [0, 0.001, 0.01, 0.1]
normalize_rewards = [False]
kl_ratios = [False]
mdps = [GymEnv("Freeway-v0")]
# mdp_classes = [MountainCarEnvX]

param_cart_product = itertools.product(
    kl_ratios, normalize_rewards, mdps, etas, seeds
)

for kl_ratio, normalize_reward, mdp, eta, seed in param_cart_product:

    network = ConvNetwork(
        input_shape=(3, 42, 32),
        output_dim=mdp.spec.action_space.n,
        hidden_sizes=(20,),
        conv_filters=(16, 16),
        conv_filter_sizes=(4, 4),
        conv_strides=(2, 2),
        conv_pads=(0, 0),
    )
    policy = CategoricalMLPPolicy(
        env_spec=mdp.spec,
        prob_network=network,
    )

#     policy = CategoricalMLPPolicy(
#         env_spec=mdp.spec,
#         hidden_sizes=(64, 64)
#     )

    baseline = LinearFeatureBaseline(
        mdp.spec
    )

    deconv_filters = 16
    filter_sizes = 5

    algo = TRPO(
        # TRPO settings
        # -------------
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=500,
        whole_paths=True,
        max_path_length=10,
        n_itr=100,
        step_size=0.01,
        subsample_factor=1.0,
        # -------------

        # VIME settings
        # -------------
        eta=eta,
        snn_n_samples=1,
        use_replay_pool=True,
        use_kl_ratio=kl_ratio,
        use_kl_ratio_q=kl_ratio,
        kl_batch_size=4,
        normalize_reward=normalize_reward,
        replay_pool_size=10000,
        n_updates_per_sample=500,
        second_order_update=True,
        layers_disc=[
            dict(name='input', in_shape=(3, 42, 32)),
            dict(name='convolution', n_filters=16,
                 filter_size=(filter_sizes, filter_sizes), stride=(1, 1)),
            dict(name='convolution', n_filters=16,
                 filter_size=(4, 4), stride=(2, 2)),
            dict(name='convolution', n_filters=16,
                 filter_size=(filter_sizes, filter_sizes), stride=(1, 1)),
            dict(name='reshape', shape=([0], -1)),
            dict(name='gaussian', n_units=2016),
            dict(name='gaussian', n_units=128),
            dict(name='gaussian', n_units=2016),
            dict(name='reshape', shape=([0], 16, 14, 9)),
            dict(name='deconvolution', n_filters=deconv_filters,
                 filter_size=(filter_sizes, filter_sizes), stride=(1, 1)),
            dict(name='deconvolution', n_filters=deconv_filters,
                 filter_size=(4, 4), stride=(2, 2)),
            dict(name='deconvolution', n_filters=3,
                 filter_size=(filter_sizes, filter_sizes), stride=(1, 1)),
        ],
        unn_learning_rate=0.001,
        surprise_transform=BatchPolopt.SurpriseTransform.CAP90PERC,
        update_likelihood_sd=True,
        replay_kl_schedule=0.99,
        output_type=BNN.OutputType.REGRESSION,
        pool_batch_size=2,
        likelihood_sd_init=1.0,
        prior_sd=0.5,
        # -------------
        disable_variance=False,
        group_variance_by=BNN.GroupVarianceBy.WEIGHT,
        surprise_type=BNN.SurpriseType.INFGAIN,
        predict_reward=True,
        use_local_reparametrization_trick=True,
        n_itr_update=1,
        # -------------
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-vime-freeway-i",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        dry=False,
        use_gpu=True,
        script="sandbox/rein/experiments/run_experiment_lite.py",
    )
