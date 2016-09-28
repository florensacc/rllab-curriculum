from sandbox.rocky.neural_learner.envs.doom_goal_finding_maze_env import DoomGoalFindingMazeEnv
from sandbox.rocky.neural_learner.algos.pposgd_clip_ratio import PPOSGD
from sandbox.rocky.tf.policies.categorical_rnn_policy import CategoricalRNNPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.misc.instrument import stub, run_experiment_lite
import tensorflow as tf

from sandbox.rocky.tf.policies.rnn_utils import NetworkType
from sandbox.rocky.tf.core.network import ConvNetwork
from rllab import config

# stub(globals())

env = TfEnv(DoomGoalFindingMazeEnv())
baseline = ZeroBaseline(env_spec=env.spec)
policy = CategoricalRNNPolicy(
    env_spec=env.spec,
    hidden_nonlinearity=tf.nn.relu,
    weight_normalization=True,
    layer_normalization=False,
    network_type=NetworkType.GRU,
    state_include_action=False,
    feature_network=ConvNetwork(
        name="embedding_network",
        input_shape=env.observation_space.shape,
        output_dim=256,
        hidden_sizes=(),
        conv_filters=(16, 32),
        conv_filter_sizes=(5, 5),
        conv_strides=(4, 2),
        conv_pads=('VALID', 'VALID'),
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.relu,
        weight_normalization=True,
        batch_normalization=False,
    ),
    name="policy"
)

algo = PPOSGD(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=10000,
    max_path_length=500,
    sampler_args=dict(n_envs=20),
    n_steps=40,
    minibatch_size=256,
    n_epochs=3,
    n_itr=1000,
    clip_lr=0.2,
    log_loss_kl_before=False,
    log_loss_kl_after=False,
    # n_epochs=0,
)

USE_GPU = False  # True#False

if USE_GPU:
    config.DOCKER_IMAGE = "dementrock/rllab3-vizdoom-gpu-cuda80"
else:
    config.DOCKER_IMAGE = "dementrock/rllab3-vizdoom"

run_experiment_lite(
    algo.train(),
    exp_prefix="doom_maze_1",
    mode="local",
    n_parallel=0,
    seed=11,
    use_gpu=USE_GPU,
    # variant=v,
    snapshot_mode="last",
    # env=dict(CUDA_VISIBLE_DEVICES="0")
)
