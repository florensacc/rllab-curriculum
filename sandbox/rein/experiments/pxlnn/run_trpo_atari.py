import os
import tensorflow as tf
import itertools

from sandbox.rein.algos.pxlnn.trpo_plus import TRPOPlus
from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.core.network import ConvNetwork
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rein.dynamics_models.tf_autoenc.autoenc import ConvAutoEncoder

from sandbox.rein.envs.tf_atari import AtariEnv

os.environ["THEANO_FLAGS"] = "device=gpu"

stub(globals())

TEST_RUN = True

# global params
num_seq_frames = 1
batch_norm = True
dropout = False

# Param ranges
if TEST_RUN:
    exp_prefix = 'bin-count-nonoise-c'
    seeds = range(1)
    etas = [0.1]
    mdps = [AtariEnv(game='frostbite', obs_type="image", frame_skip=8),
            AtariEnv(game='freeway', obs_type="image", frame_skip=8),
            AtariEnv(game='montezuma_revenge', obs_type="image", frame_skip=8)]
    lst_factor = [1]
    trpo_batch_size = 1000
    max_path_length = 450
    batch_norm = True
else:
    exp_prefix = 'trpo-count-atari-42x52-a'
    seeds = range(5)
    etas = [0, 1.0, 0.1, 0.01]
    mdps = [AtariEnv(game='frostbite', obs_type="image", frame_skip=8),
            AtariEnv(game='montezuma_revenge', obs_type="image", frame_skip=8),
            AtariEnv(game='freeway', obs_type="image", frame_skip=8)]
    lst_factor = [1]
    trpo_batch_size = 20000
    max_path_length = 4500
    batch_norm = True

param_cart_product = itertools.product(
    lst_factor, mdps, etas, seeds
)

for factor, mdp, eta, seed in param_cart_product:
    network = ConvNetwork(
        name='policy_network',
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.softmax,
        input_shape=(mdp.spec.observation_space.shape[2], mdp.spec.observation_space.shape[1], num_seq_frames),
        output_dim=mdp.spec.action_space.flat_dim,
        hidden_sizes=(64,),
        conv_filters=(16, 16),
        conv_filter_sizes=(6, 6),
        conv_strides=(2, 2),
        conv_pads=('VALID', 'VALID'),
    )
    policy = CategoricalMLPPolicy(
        name='policy',
        env_spec=mdp.spec,
        prob_network=network,
    )

    network = ConvNetwork(
        name='baseline_network',
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.identity,
        input_shape=(mdp.spec.observation_space.shape[2], mdp.spec.observation_space.shape[1], num_seq_frames),
        output_dim=1,
        hidden_sizes=(32,),
        conv_filters=(16, 16),
        conv_filter_sizes=(6, 6),
        conv_strides=(2, 2),
        conv_pads=('VALID', 'VALID'),
    )
    baseline = GaussianMLPBaseline(
        env_spec=mdp.spec,
    )

    # Dynamics model f: num_seq_frames x h x w -> h x w
    dynamics_model = ConvAutoEncoder(
        input_shape=(None, mdp.spec.observation_space.shape[1], mdp.spec.observation_space.shape[2], num_seq_frames),
        n_filters=[num_seq_frames, 10, 10],
        filter_sizes=[6, 6, 6],
    )

    algo = TRPOPlus(
        dynamics_model=dynamics_model,
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=trpo_batch_size,
        max_path_length=max_path_length,
        n_itr=400,
        step_size=0.01,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_prefix,
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        dry=False,
        use_gpu=True,
        script="sandbox/rein/experiments/run_experiment_lite_count.py",
        sync_all_data_node_to_s3=True
    )
