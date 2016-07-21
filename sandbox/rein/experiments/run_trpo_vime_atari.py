from sandbox.rein.algos.trpo_vime import TRPO
import os
from sandbox.rein.envs.mountain_car_env_x import MountainCarEnvX
from sandbox.rein.envs.double_pendulum_env_x import DoublePendulumEnvX
from sandbox.rein.envs.gym_env_downscaled import GymEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rein.dynamics_models.bnn.bnn import BNN
from rllab.core.network import ConvNetwork
import lasagne

from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.misc.instrument import stub, run_experiment_lite
import itertools
from sandbox.rein.algos.batch_polopt_vime import BatchPolopt
os.environ["THEANO_FLAGS"] = "device=gpu"

stub(globals())

# Param ranges
seeds = range(5)
etas = [0, 0.01, 0.1]
# seeds = range(5)
# etas = [0, 0.001, 0.01, 0.1]
normalize_rewards = [False]
kl_ratios = [True]
mdps = [GymEnv("Freeway-v0", record_video=False)]

param_cart_product = itertools.product(
    kl_ratios, normalize_rewards, mdps, etas, seeds
)

for kl_ratio, normalize_reward, mdp, eta, seed in param_cart_product:

    network = ConvNetwork(
        input_shape=mdp.spec.observation_space.shape,
        output_dim=mdp.spec.action_space.flat_dim,
        hidden_sizes=(32,),
        conv_filters=(16, 16, 16),
        conv_filter_sizes=(6, 6, 6),
        conv_strides=(2, 2, 2),
        conv_pads=(0, 2, 2),
    )
    policy = CategoricalMLPPolicy(
        env_spec=mdp.spec,
        prob_network=network,
    )

    network = ConvNetwork(
        input_shape=mdp.spec.observation_space.shape,
        output_dim=1,
        hidden_sizes=(32,),
        conv_filters=(16, 16, 16),
        conv_filter_sizes=(6, 6, 6),
        conv_strides=(2, 2, 2),
        conv_pads=(0, 2, 2),
    )
    baseline = GaussianMLPBaseline(
        mdp.spec,
        regressor_args=dict(mean_network=network,
                            batchsize=50),
    )

    algo = TRPO(
        # TRPO settings
        # -------------
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=100,
        whole_paths=True,
        max_path_length=10,
        n_itr=500,
        step_size=0.01,
        optimizer_args=dict(num_slices=10),
        # -------------

        # VIME settings
        # -------------
        eta=eta,
        snn_n_samples=1,
        use_replay_pool=True,
        pool_args=dict(subsample_factor=0.1),
        use_kl_ratio=kl_ratio,
        use_kl_ratio_q=kl_ratio,
        kl_batch_size=32,
        normalize_reward=normalize_reward,
        replay_pool_size=1000,
        n_updates_per_sample=100,
        second_order_update=True,
        state_dim=mdp.spec.observation_space.shape,
        action_dim=(mdp.spec.action_space.flat_dim,),
        reward_dim=(1,),
        layers_disc=[
            dict(name='convolution',
                 n_filters=16,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0)),
            dict(name='convolution',
                 n_filters=16,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2)),
            dict(name='convolution',
                 n_filters=16,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2)),
            dict(name='reshape',
                 shape=([0], -1)),
            dict(name='gaussian',
                 n_units=256),
            dict(name='gaussian',
                 n_units=512),
            dict(name='outerprod'),
            dict(name='gaussian',
                 n_units=256),
            dict(name='split',
                 n_units=128),
            dict(name='gaussian',
                 n_units=1600),
            dict(name='reshape',
                 shape=([0], 16, 10, 10)),
            dict(name='deconvolution',
                 n_filters=16,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 nonlinearity=lasagne.nonlinearities.rectify),
            dict(name='deconvolution',
                 n_filters=16,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 nonlinearity=lasagne.nonlinearities.rectify),
            dict(name='deconvolution',
                 n_filters=mdp.spec.observation_space.shape[0],
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0),
                 nonlinearity=lasagne.nonlinearities.linear),
        ],
        unn_learning_rate=0.005,
        surprise_transform=BatchPolopt.SurpriseTransform.CAP90PERC,
        update_likelihood_sd=True,
        replay_kl_schedule=0.98,
        output_type=BNN.OutputType.REGRESSION,
        pool_batch_size=32,
        likelihood_sd_init=0.1,
        prior_sd=0.05,
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
        exp_prefix="trpo-vime-freeway-pxl-c",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        dry=False,
        use_gpu=True,
        script="sandbox/rein/experiments/run_experiment_lite.py",
    )
