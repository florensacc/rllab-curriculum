import os
import lasagne
import itertools

from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.core.network import ConvNetwork
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.rein.envs.atari import AtariEnvX
from sandbox.rein.algos.trpo_count import TRPO
from sandbox.rein.algos.batch_polopt_vime import BatchPolopt
from sandbox.rein.dynamics_models.bnn.conv_bnn_vime import ConvBNNVIME

os.environ["THEANO_FLAGS"] = "device=gpu"

stub(globals())

# global params
num_seq_frames = 4
dyn_pool_enable = True
batch_norm = True
dropout = False

# Param ranges
seeds = list(range(5))
etas = [1., 0.01]
lst_factor = [1]
lst_pred_delta = [False]
kl_ratios = [False]
mdps = [AtariEnvX(game='freeway', obs_type="image", frame_skip=4),
        AtariEnvX(game='breakout', obs_type="image", frame_skip=4),
        AtariEnvX(game='frostbite', obs_type="image", frame_skip=4),
        AtariEnvX(game='montezuma_revenge', obs_type="image", frame_skip=4)]

param_cart_product = itertools.product(
    lst_pred_delta, lst_factor, kl_ratios, mdps, etas, seeds
)

for pred_delta, factor, kl_ratio, mdp, eta, seed in param_cart_product:
    network = ConvNetwork(
        input_shape=(num_seq_frames,) + (mdp.spec.observation_space.shape[1], mdp.spec.observation_space.shape[2]),
        output_dim=mdp.spec.action_space.flat_dim,
        hidden_sizes=(64,),
        conv_filters=(16, 16, 16),
        conv_filter_sizes=(6, 6, 6),
        conv_strides=(2, 2, 2),
        conv_pads=(0, 2, 2),
    )
    policy = CategoricalMLPPolicy(
        env_spec=mdp.spec,
        num_seq_inputs=num_seq_frames,
        prob_network=network,
    )

    network = ConvNetwork(
        input_shape=(num_seq_frames,) + (mdp.spec.observation_space.shape[1], mdp.spec.observation_space.shape[2]),
        output_dim=1,
        hidden_sizes=(64,),
        conv_filters=(16, 16, 16),
        conv_filter_sizes=(6, 6, 6),
        conv_strides=(2, 2, 2),
        conv_pads=(0, 2, 2),
    )
    baseline = GaussianMLPBaseline(
        mdp.spec,
        num_seq_inputs=num_seq_frames,
        regressor_args=dict(
            mean_network=network,
            batchsize=30000,
            subsample_factor=0.1),
    )

    # If we don't use a replay pool, we could have correct values here, as
    # it is purely Bayesian. We then divide the KL divergence term by the
    # number of batches in each iteration `batch'. Also the batch size
    # would be given correctly.
    if dyn_pool_enable:
        batch_size = 1
        n_batches = 50
    else:
        batch_size = 1
        n_batches = 1


    algo = TRPO(
        # TRPO settings
        # -------------
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        dyn_mdl=dyn_mdl,
        batch_size=500,
        whole_paths=True,
        max_path_length=45,
        n_itr=400,
        step_size=0.01,
        optimizer_args=dict(
            num_slices=30,
            subsample_factor=0.1),

        # VIME settings
        # -------------
        eta=eta,
        use_kl_ratio=kl_ratio,
        use_kl_ratio_q=kl_ratio,
        kl_batch_size=512,
        num_sample_updates=1,  # Every sample in traj batch will be used in `num_sample_updates' updates.
        normalize_reward=False,
        replay_kl_schedule=0.98,
        n_itr_update=1,  # Fake itr updates in sampler
        dyn_pool_args=dict(
            enable=dyn_pool_enable,
            size=100000,
            min_size=10,
            batch_size=32
        ),
        surprise_transform=None,#BatchPolopt.SurpriseTransform.CAP99PERC,
        predict_delta=pred_delta,
        num_seq_frames=num_seq_frames,
        predict_reward=True,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-vime-atari-42x52-",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        dry=False,
        use_gpu=True,
        script="sandbox/rein/experiments/run_experiment_lite.py",
    )
