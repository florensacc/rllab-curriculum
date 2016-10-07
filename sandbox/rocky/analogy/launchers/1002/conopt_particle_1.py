import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import tensorflow as tf
from rllab.misc.instrument import stub, run_experiment_lite
from rllab import config

from sandbox.rocky.tf.policies.rnn_utils import NetworkType

# stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

"""
Test conopt particle with different amount of training data
"""

USE_GPU = False  # True#False  # True#False
USE_CIRRASCALE = False  # True
MODE = "lab_kube"
SETUP = "conopt"
ENV = "particle-2"


# SETUP = "simple"


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]

    @variant
    def n_train_trajs(self):
        return [300, 1000, 5000]

    @variant
    def use_shuffler(self):
        return [False, True]#, False]#, True]

    @variant
    def state_include_action(self):
        return [False, True]#, False]#, True]

    @variant
    def horizon(self):
        return [30]#, 100]

    @variant
    def n_epochs(self):
        return [1000]

    @variant
    def obs_type(self):
        return ["full_state"]#, "image"]

    @variant(hide=True)
    def demo_cache_key(self, horizon, obs_type):
        yield "%s-%s-%s-%s-v3" % (
            SETUP,
            ENV,
            str(horizon),
            obs_type.replace("_", "-")
        ),


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))


def run_task(v):
    from sandbox.rocky.analogy.algos.trainer import Trainer
    from sandbox.rocky.analogy.demo_collector.policy_demo_collector import PolicyDemoCollector
    from sandbox.rocky.analogy.demo_collector.trajopt_demo_collector import TrajoptDemoCollector
    from sandbox.rocky.analogy.envs.conopt_particle_env import ConoptParticleEnv
    from sandbox.rocky.analogy.envs.simple_particle_env import SimpleParticleEnv
    from sandbox.rocky.analogy.policies.conopt_particle_tracking_policy import ConoptParticleTrackingPolicy
    from sandbox.rocky.analogy.policies.simple_particle_tracking_policy import SimpleParticleTrackingPolicy
    from sandbox.rocky.analogy.policies.demo_rnn_mlp_analogy_policy import DemoRNNMLPAnalogyPolicy
    from sandbox.rocky.tf.envs.base import TfEnv

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
            ConoptParticleEnv if SETUP == "conopt" else
            SimpleParticleEnv,
            obs_type=v["obs_type"],
        ),
        demo_collector=PolicyDemoCollector(
            policy_cls=ConoptParticleTrackingPolicy if SETUP == "conopt" else
            SimpleParticleTrackingPolicy
        ),
        demo_cache_key=v["demo_cache_key"],
        n_train_trajs=v["n_train_trajs"],
        n_test_trajs=50,
        horizon=v["horizon"],
        n_epochs=v["n_epochs"],
        learning_rate=1e-2,
        no_improvement_tolerance=20,
        shuffler=ConoptParticleEnv.shuffler() if v["use_shuffler"] else None,
        batch_size=100,
    )
    algo.train()


# First, make sure that all

for v in variants:

    config.DOCKER_IMAGE = "dementrock/rllab3-conopt-gpu"
    config.KUBE_DEFAULT_NODE_SELECTOR = {
    }
    config.KUBE_DEFAULT_RESOURCES = {
        "requests": {
            "cpu": 8,
            "memory": "10Gi",
        },
    }

    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="conopt-particle-1-2",
        mode=MODE,
        n_parallel=0,
        seed=v["seed"],
        snapshot_mode="last",
        variant=v,
        terminate_machine=True,
        sync_all_data_node_to_s3=False,
    )
    # sys.exit()
