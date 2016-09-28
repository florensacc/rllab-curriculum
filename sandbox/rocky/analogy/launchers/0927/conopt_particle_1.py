from sandbox.rocky.analogy.algos.trainer import Trainer
from sandbox.rocky.analogy.envs.conopt_particle_env import ConoptParticleEnv
from sandbox.rocky.analogy.policies.conopt_particle_tracking_policy import ConoptParticleTrackingPolicy
from sandbox.rocky.analogy.policies.demo_rnn_mlp_analogy_policy import DemoRNNMLPAnalogyPolicy
from sandbox.rocky.tf.envs.base import TfEnv
import tensorflow as tf
from rllab.misc.instrument import run_experiment_lite

from sandbox.rocky.tf.policies.rnn_utils import NetworkType

env = TfEnv(ConoptParticleEnv(
    seed=0,
))

policy = DemoRNNMLPAnalogyPolicy(
    env_spec=env.spec,
    name="policy",
    rnn_hidden_size=100,
    rnn_hidden_nonlinearity=tf.nn.relu,
    mlp_hidden_sizes=(100, 100),
    mlp_hidden_nonlinearity=tf.nn.relu,
    state_include_action=True,
    batch_normalization=False,
    layer_normalization=False,
    weight_normalization=True,
    network_type=NetworkType.GRU,
)

algo = Trainer(
    policy=policy,
    env_cls=TfEnv.wrap(
        ConoptParticleEnv,
    ),
    demo_policy_cls=ConoptParticleTrackingPolicy,
    n_train_trajs=50,
    n_test_trajs=20,
    horizon=20,
    n_epochs=1000,
    learning_rate=1e-2,
    no_improvement_tolerance=20,
    shuffler=None,
    batch_size=100,
)

run_experiment_lite(
    algo.train(),
    exp_prefix="conopt-particle-1",
    mode="local",
    n_parallel=0,
    seed=11,
    snapshot_mode="last",
)
