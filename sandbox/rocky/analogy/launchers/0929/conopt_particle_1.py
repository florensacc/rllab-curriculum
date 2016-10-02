from sandbox.rocky.analogy.algos.trainer import Trainer
from sandbox.rocky.analogy.demo_collector.policy_demo_collector import PolicyDemoCollector
from sandbox.rocky.analogy.demo_collector.trajopt_demo_collector import TrajoptDemoCollector
from sandbox.rocky.analogy.envs.conopt_particle_env import ConoptParticleEnv
from sandbox.rocky.analogy.policies.conopt_particle_tracking_policy import ConoptParticleTrackingPolicy
from sandbox.rocky.analogy.policies.demo_rnn_mlp_analogy_policy import DemoRNNMLPAnalogyPolicy
from sandbox.rocky.tf.envs.base import TfEnv


import tensorflow as tf
from rllab.misc.instrument import stub, run_experiment_lite
from rllab import config

from sandbox.rocky.tf.policies.rnn_utils import NetworkType

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

"""
Try compressed image representation
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]

    @variant
    def n_train_trajs(self):
        return [300, 1000, 5000]

    @variant
    def use_shuffler(self):
        return [False, True]

    @variant
    def state_include_action(self):
        return [False, True]

    @variant
    def horizon(self):
        return [30, 100]


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
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
        state_include_action=v["state_include_action"],
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
        demo_collector=TrajoptDemoCollector(),
        # demo_collector=PolicyDemoCollector(policy_cls=ConoptParticleTrackingPolicy),
        n_train_trajs=v["n_train_trajs"],
        n_test_trajs=50,
        horizon=v["horizon"],
        n_epochs=1000,
        learning_rate=1e-2,
        no_improvement_tolerance=20,
        shuffler=ConoptParticleEnv.shuffler() if v["use_shuffler"] else None,
        batch_size=100,
    )

    config.DOCKER_IMAGE = "dementrock/rllab3-conopt-gpu"
    config.KUBE_DEFAULT_NODE_SELECTOR = {
        # "aws/type": "c4.2xlarge",
    }

    # config.KUBE_DEFAULT_NODE_SELECTOR = {
    #     "aws/type": "g2.2xlarge",
    # }
    # config.USE_GPU = True
    # config.DOCKER_IMAGE = "dementrock/rllab3-shared-gpu"
    config.KUBE_DEFAULT_RESOURCES = {
        "requests": {
            "cpu": 4 * 0.75,
        },
        "limits": {
            "cpu": 4 * 0.75,
        },
    }

    run_experiment_lite(
        algo.train(),
        exp_prefix="conopt-particle-1",
        mode="local",
        n_parallel=8,
        seed=v["seed"],
        snapshot_mode="last",
        variant=v,
        # dry=True
    )
    # sys.exit()
