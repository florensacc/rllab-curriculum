from sandbox.rocky.analogy.envs.simple_particle_env import SimpleParticleEnv
from sandbox.rocky.analogy.policies.simple_particle_tracking_policy import SimpleParticleTrackingPolicy
from sandbox.rocky.analogy.policies.double_lstm_policy import DoubleLSTMPolicy
from sandbox.rocky.analogy.policies.demo_rnn_mlp_analogy_policy import DemoRNNMLPAnalogyPolicy
from sandbox.rocky.analogy.policies.mlp_analogy_policy import MLPAnalogyPolicy
from sandbox.rocky.analogy.algos.trainer import Trainer
from sandbox.rocky.tf.envs.base import TfEnv
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.layers import OrthogonalInitializer, XavierUniformInitializer
import tensorflow as tf
import math
from rllab.misc.instrument import stub, run_experiment_lite
import sys

from sandbox.rocky.tf.policies import rnn_utils

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

"""
Entry point for Bradly
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31, 41, 51]

    @variant
    def n_particles(self):
        return [2]  # 5]#3, 4, 5, 6]

    @variant
    def n_train_trajs(self):
        return [1000]#, 20000]  # , 20000]#1000, 5000, 20000]

    @variant
    def hidden_dim(self):
        return [100]#50, 100]

    @variant
    def use_shuffler(self):
        return [True]  # True, False]

    @variant
    def batch_size(self):
        return [100]  # 10, 100]

    @variant
    def network_type(self):
        return [
            # rnn_utils.NetworkType.PSEUDO_LSTM,
            # rnn_utils.NetworkType.LSTM,
            # rnn_utils.NetworkType.LSTM_PEEPHOLE,
            rnn_utils.NetworkType.GRU,
            # # rnn_utils.NetworkType.TF_GRU,
            # # rnn_utils.NetworkType.TF_BASIC_LSTM,
            # rnn_utils.NetworkType.PSEUDO_LSTM_GATE_SQUASH,
        ]

    @variant
    def nonlinearity(self):
        return ["relu"]#, "tanh"]

    @variant
    def ortho_init(self):
        return [True]#True, False]

    @variant
    def layer_normalization(self):
        return [False]#True, False]#True, False]

    @variant
    def weight_normalization(self):
        return [True]#, False]

    @variant
    def min_margin(self):
        return [True]#, False]


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    env = TfEnv(SimpleParticleEnv(seed=0, n_particles=v["n_particles"]))
    policy = DemoRNNMLPAnalogyPolicy(
        env_spec=env.spec,
        name="policy",
        rnn_hidden_size=v["hidden_dim"],
        rnn_hidden_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
        mlp_hidden_sizes=(v["hidden_dim"], v["hidden_dim"]),
        mlp_hidden_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
        state_include_action=True,
        network_type=v["network_type"],
        layer_normalization=v["layer_normalization"],
        weight_normalization=v["weight_normalization"],
        network_args=dict(
            W_h_init=OrthogonalInitializer() if v["ortho_init"] else XavierUniformInitializer(),
            # fixed_horizon=20,
        )
    )
    algo = Trainer(
        policy=policy,
        env_cls=TfEnv.wrap(
            SimpleParticleEnv,
            n_particles=v["n_particles"],
            min_margin=(((0.8 * 2) ** 2 / v["n_particles"]) ** 0.5) / 2 if v["min_margin"] else 0,
            min_angular_margin=math.pi / v["n_particles"] if v["min_margin"] else 0
        ),
        demo_policy_cls=SimpleParticleTrackingPolicy,
        n_train_trajs=v["n_train_trajs"],
        n_test_trajs=50,
        horizon=20,
        n_epochs=1000,
        learning_rate=1e-2,
        no_improvement_tolerance=10,
        shuffler=SimpleParticleEnv.shuffler() if v["use_shuffler"] else None,
        batch_size=v["batch_size"],
        # plot=True,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="analogy-particle-22",
        mode="local",
        n_parallel=4,
        seed=v["seed"],
        variant=v,
        snapshot_mode="last",
    )
    sys.exit()
