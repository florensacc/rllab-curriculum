from sandbox.bradly.daggar_feedback.trainer.trainer import Trainer
from sandbox.bradly.daggar_feedback.demo_collector.policy_demo_collector import MixtureDemoCollector
from sandbox.rocky.analogy.demo_collector.trajopt_demo_collector import TrajoptDemoCollector
from sandbox.bradly.supervised_woj.envs.conopt_particle_env import ConoptParticleEnv
from sandbox.bradly.supervised_woj.policy.conopt_particle_tracking_policy import ConoptParticleTrackingPolicy
from sandbox.bradly.daggar_feedback.policy.mlp_policy import MLPPolicy
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
        return [11]

    @variant
    def n_train_trajs(self):
        return [50]

    @variant
    def use_shuffler(self):
        return [False]

    @variant
    def state_include_action(self):
        return [False]

    @variant
    def horizon(self):
        return [100]


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    env = TfEnv(ConoptParticleEnv(
        seed=0
    ))

    policy = MLPPolicy(
        env_spec=env.spec,
    )

    algo = Trainer(
        policy=policy,
        env_cls=TfEnv.wrap(
            ConoptParticleEnv,
        ),
        #demo_collector=TrajoptDemoCollector(),
        demo_collector=MixtureDemoCollector(policy_cls=ConoptParticleTrackingPolicy),
        n_train_trajs=v["n_train_trajs"],
        n_test_trajs=50,
        horizon=v["horizon"],
        n_epochs=1000,
        learning_rate=1e-2,
        batch_size=100,
        plot=True
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
        n_parallel=4,
        seed=v["seed"],
        snapshot_mode="last",
        variant=v,
        # dry=True
    )
    # sys.exit()
