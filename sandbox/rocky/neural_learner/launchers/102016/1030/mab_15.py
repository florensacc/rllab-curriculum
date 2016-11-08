import random

from rllab.misc.instrument import run_experiment_lite
from rllab import config
from rllab.misc.instrument import VariantGenerator, variant
import numpy as np

from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.neural_learner.envs.mab_env import MABEnv
from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv

"""
Test resetter - bug fixes
"""

USE_GPU = True
USE_CIRRASCALE = False
# MODE = "local"
MODE = launch_cirrascale("pascal")


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31, 41, 51]

    @variant
    def batch_size(self):
        if MODE == "local":
            return [10000]
        return [250000]

    @variant
    def docker_image(self):
        return [
            "dementrock/rllab3-vizdoom-gpu-cuda80:cig",
        ]

    @variant
    def clip_lr(self):
        return [0.1]

    @variant
    def use_kl_penalty(self):
        return [False]

    @variant
    def nonlinearity(self):
        return ["relu"]

    @variant
    def n_arms(self):
        return [5, 50]

    @variant
    def mean_kl(self):
        return [0.01]

    @variant
    def layer_normalization(self):
        return [False]

    @variant
    def n_episodes(self):
        return [500]

    @variant
    def weight_normalization(self):
        return [True]

    @variant
    def min_epochs(self):
        if MODE == "local":
            return [1]
        return [5]

    @variant
    def opt_batch_size(self):
        return [128]

    @variant
    def opt_n_steps(self):
        return [None]

    @variant
    def batch_normalization(self):
        return [False]

    @variant
    def entropy_bonus_coeff(self):
        return [0]

    @variant
    def discount(self):
        return [0.99]

    @variant
    def gae_lambda(self):
        return [0.3]

    @variant
    def hidden_dim(self):
        return [256]

    @variant
    def use_ac(self):
        return [True]  # , False]#, True]

    @variant
    def use_reset(self):
        return [True, False]  # False, True]

    @variant
    def reset_ratio(self, use_reset):
        if not use_reset:
            yield None
        else:
            yield 0.7
            yield 0.5
            yield 0.2
            yield 0.


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for vv in variants:

    def run_task(v):
        from sandbox.rocky.neural_learner.baselines.l2_rnn_baseline import L2RNNBaseline
        from sandbox.rocky.neural_learner.algos.pposgd_clip_ratio import PPOSGD
        from sandbox.rocky.neural_learner.algos.pposgd_joint_ac import PPOSGD as PPOSGDAC
        from sandbox.rocky.neural_learner.optimizers.tbptt_optimizer import TBPTTOptimizer
        from sandbox.rocky.neural_learner.policies.categorical_rnn_policy import CategoricalRNNPolicy
        from sandbox.rocky.tf.envs.base import TfEnv
        import tensorflow as tf
        from sandbox.rocky.tf.policies.rnn_utils import NetworkType

        if v["use_reset"]:
            from sandbox.rocky.neural_learner.envs.multi_resetter_env import MultiResetterEnv
            env = TfEnv(MultiResetterEnv(
                wrapped_env=MABEnv(n_arms=v["n_arms"]),
                n_episodes=v["n_episodes"],
                episode_horizon=1,
                discount=v["discount"],
                reset_ratio=v["reset_ratio"]
            ))
        else:
            env = TfEnv(MultiEnv(
                wrapped_env=MABEnv(n_arms=v["n_arms"]),
                n_episodes=v["n_episodes"],
                episode_horizon=1,
                discount=v["discount"]
            ))

        if v["use_ac"]:
            from sandbox.rocky.neural_learner.policies.categorical_rnn_actor_critic import CategoricalRNNActorCritic
            ac = CategoricalRNNActorCritic(
                name="ac",
                env_spec=env.spec,
                hidden_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
                weight_normalization=v["weight_normalization"],
                layer_normalization=v["layer_normalization"],
                network_type=NetworkType.GRU,
                hidden_dim=v["hidden_dim"],
                record_prev_states=v["use_reset"],
            )
            baseline = ac
            policy = ac
            algo = PPOSGDAC(
                env=env,
                # ac=ac,
                policy=policy,
                baseline=baseline,
                batch_size=v["batch_size"],
                max_path_length=v["n_episodes"],
                sampler_args=dict(n_envs=max(1, int(np.ceil(v["batch_size"] / v["n_episodes"])))),
                n_itr=10000,
                step_size=v["mean_kl"],
                clip_lr=v["clip_lr"],
                log_loss_kl_before=False,
                log_loss_kl_after=False,
                use_kl_penalty=v["use_kl_penalty"],
                min_n_epochs=v["min_epochs"],
                entropy_bonus_coeff=v["entropy_bonus_coeff"],
                optimizer=TBPTTOptimizer(
                    batch_size=v["opt_batch_size"],
                    n_steps=v["opt_n_steps"],
                    n_epochs=v["min_epochs"],
                ),
                discount=v["discount"],
                gae_lambda=v["gae_lambda"],
                use_line_search=True
            )
        else:
            assert not v["use_reset"]
            baseline = L2RNNBaseline(
                name="vf",
                env_spec=env.spec,
                log_loss_before=False,
                log_loss_after=False,
                hidden_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
                weight_normalization=v["weight_normalization"],
                layer_normalization=v["layer_normalization"],
                state_include_action=False,
                hidden_dim=v["hidden_dim"],
                optimizer=TBPTTOptimizer(
                    batch_size=v["opt_batch_size"],
                    n_steps=v["opt_n_steps"],
                    n_epochs=v["min_epochs"],
                ),
                batch_size=v["opt_batch_size"],
                n_steps=v["opt_n_steps"],
            )
            policy = CategoricalRNNPolicy(
                env_spec=env.spec,
                hidden_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
                weight_normalization=v["weight_normalization"],
                layer_normalization=v["layer_normalization"],
                network_type=NetworkType.GRU,
                hidden_dim=v["hidden_dim"],
                # state_include_action=True,
                name="policy",
            )

            algo = PPOSGD(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=v["batch_size"],
                max_path_length=v["n_episodes"],
                sampler_args=dict(n_envs=max(1, int(np.ceil(v["batch_size"] / v["n_episodes"])))),
                n_itr=10000,
                step_size=v["mean_kl"],
                clip_lr=v["clip_lr"],
                log_loss_kl_before=False,
                log_loss_kl_after=False,
                use_kl_penalty=v["use_kl_penalty"],
                min_n_epochs=v["min_epochs"],
                entropy_bonus_coeff=v["entropy_bonus_coeff"],
                optimizer=TBPTTOptimizer(
                    batch_size=v["opt_batch_size"],
                    n_steps=v["opt_n_steps"],
                    n_epochs=v["min_epochs"],
                ),
                discount=v["discount"],
                gae_lambda=v["gae_lambda"],
                use_line_search=True
            )
        algo.train()


    config.DOCKER_IMAGE = vv["docker_image"]  # "dementrock/rllab3-vizdoom-gpu-cuda80"
    config.KUBE_DEFAULT_NODE_SELECTOR = {
        "aws/type": "c4.8xlarge",
    }
    config.KUBE_DEFAULT_RESOURCES = {
        "requests": {
            "cpu": 36 * 0.75,
            "memory": "50Gi",
        },
    }
    config.AWS_INSTANCE_TYPE = "m4.2xlarge"
    config.AWS_SPOT = True
    config.AWS_SPOT_PRICE = '1.0'
    config.AWS_REGION_NAME = random.choice(['us-west-2', 'us-west-1', 'us-east-1'])
    config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]
    config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[config.AWS_REGION_NAME]
    config.AWS_SECURITY_GROUP_IDS = config.ALL_REGION_AWS_SECURITY_GROUP_IDS[config.AWS_REGION_NAME]

    if MODE == "local_docker":
        env = dict(CUDA_VISIBLE_DEVICES="3")
    else:
        env = dict()

    run_experiment_lite(
        run_task,
        exp_prefix="mab-15-2",
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
