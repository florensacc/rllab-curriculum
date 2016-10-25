from rllab.misc.instrument import run_experiment_lite
from rllab import config
from rllab.misc.instrument import VariantGenerator, variant
import numpy as np
import sys
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
# from sandbox.rocky.neural_learner.policies.modular_policy import ModularPolicy
from sandbox.rocky.neural_learner.algos.admm_trainer import ADMMTrainer

"""
Doom without curriculum
"""

USE_GPU = True#False#True
USE_CIRRASCALE = False#True
# MODE = "local_docker"
MODE = launch_cirrascale("pascal")
VERSION = "v3"


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]

    @variant
    def batch_size(self):
        if MODE == "local":
            return [10000]
        return [250000]#, 1000000]

    @variant
    def docker_image(self):
        return [
            # "dementrock/rllab3-shared",
            "dementrock/rllab3-vizdoom-gpu-cuda80:cig",
        ]

    @variant
    def n_arms(self):
        return [10, 50]#2, 5, 10, 50]

    @variant
    def n_episodes(self):
        return [500]

    @variant(hide=True)
    def cache_key(self, n_arms, n_episodes, batch_size):
        yield "_".join([
            "n_arms_{0}".format(n_arms),
            "n_episodes_{0}".format(n_episodes),
            "batch_size_{0}".format(batch_size),
            VERSION,
        ])

    @variant
    def nonlinearity(self):
        yield "relu"

    @variant
    def weight_normalization(self):
        yield True

    @variant
    def layer_normalization(self):
        yield False

    @variant
    def hidden_dim(self):
        return [512]#, 256, 128, 64, 32]#32, 64, 128, 256, 512]

    @variant
    def lagrangian_penalty(self):
        return [1., 0.1, 10., 0.01]

    @variant
    def n_itr(self):
        return [500]



vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for vv in variants:

    def run_task(v):
        from sandbox.rocky.neural_learner.policies.categorical_rnn_policy import CategoricalRNNPolicy
        from sandbox.rocky.tf.envs.base import TfEnv
        from sandbox.rocky.neural_learner.policies.approximate_gittins_policy import ApproximateGittinsPolicy
        from sandbox.rocky.neural_learner.envs.mab_env import MABEnv
        from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv
        from sandbox.rocky.neural_learner.optimizers.tbptt_optimizer import TBPTTOptimizer
        import tensorflow as tf
        from sandbox.rocky.tf.policies.rnn_utils import NetworkType

        env = TfEnv(
            MultiEnv(
                wrapped_env=MABEnv(
                    n_arms=v["n_arms"]
                ),
                n_episodes=v["n_episodes"],
                episode_horizon=1,
                discount=0.99
            )
        )

        policy = CategoricalRNNPolicy(
            env_spec=env.spec,
            hidden_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
            weight_normalization=v["weight_normalization"],
            layer_normalization=v["layer_normalization"],
            network_type=NetworkType.GRU,
            hidden_dim=v["hidden_dim"],
            name="policy",
            state_include_action=False
        )

        algo = ADMMTrainer(
            env=env,
            policy=policy,
            batch_size=v["batch_size"],
            eval_batch_size=250000,
            max_path_length=v["n_episodes"],
            cache_key=v["cache_key"],
            n_itr=v["n_itr"],
            lagrangian_penalty=v["lagrangian_penalty"],
            optimizer=TBPTTOptimizer(
                n_epochs=5,
                batch_size=64,
                n_steps=None,
                anneal_learning_rate=True,
                no_improvement_tolerance=10,
            )
        )
        algo.train()


    config.DOCKER_IMAGE = vv["docker_image"]
    config.KUBE_DEFAULT_NODE_SELECTOR = {
        "aws/type": "c4.8xlarge",
    }
    config.KUBE_DEFAULT_RESOURCES = {
        "requests": {
            "cpu": 36 * 0.75,
            "memory": "50Gi",
        },
    }
    config.AWS_INSTANCE_TYPE = "c4.2xlarge"
    config.AWS_SPOT = True
    config.AWS_SPOT_PRICE = '1.0'
    config.AWS_REGION_NAME = 'us-west-1'
    config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]
    config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[config.AWS_REGION_NAME]
    config.AWS_SECURITY_GROUP_IDS = config.ALL_REGION_AWS_SECURITY_GROUP_IDS[config.AWS_REGION_NAME]

    if MODE == "local_docker":
        env = dict(CUDA_VISIBLE_DEVICES="3")
    else:
        env = dict()

    run_experiment_lite(
        run_task,
        exp_prefix="admm-mab-1",
        mode=MODE,
        n_parallel=0,
        seed=vv["seed"],
        use_gpu=USE_GPU,
        use_cloudpickle=True,
        variant=vv,
        snapshot_mode="last",
        env=env,
        terminate_machine=True,
        sync_all_data_node_to_s3=False,
    )
    # sys.exit()
