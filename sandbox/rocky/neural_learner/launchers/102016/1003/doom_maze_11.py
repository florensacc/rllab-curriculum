from rllab.misc.instrument import run_experiment_lite
from rllab import config
from rllab.misc.instrument import VariantGenerator, variant
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
import numpy as np

"""
Try cirrascale
"""

USE_GPU = True#False  # True#False  # True#False
USE_CIRRASCALE = True#False  # True
MODE = launch_cirrascale#(#)"lab_kube"
OPT_BATCH_SIZE = 256
OPT_N_STEPS = 32


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]#, 41, 51]

    @variant
    def horizon_schedule_span(self):
        return [100]#0, 100, 500]

    @variant
    def batch_size(self):
        if MODE == "local":
            return [2000]
        return [10000]#, 50000]

    @variant
    def clip_lr(self):
        return [0.1, 0.3, 1.0]  # , 0.2]

    @variant
    def use_kl_penalty(self):
        return [False, True]#, False]#0.1, 0.2, 0.05]

    @variant
    def nonlinearity(self):
        return ["relu"]  # , "tanh"]

    @variant
    def layer_normalization(self):
        return [False]  # True]#, False]

    @variant
    def weight_normalization(self):
        return [True]#, False]  # True]#, False]

    @variant
    def batch_normalization(self):
        return [False]  # True, False]


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for vv in variants:

    def run_task(v):
        from sandbox.rocky.neural_learner.baselines.l2_rnn_baseline import L2RNNBaseline
        from sandbox.rocky.neural_learner.envs.doom_goal_finding_maze_env import DoomGoalFindingMazeEnv
        from sandbox.rocky.neural_learner.algos.pposgd_clip_ratio import PPOSGD
        from sandbox.rocky.neural_learner.optimizers.tbptt_optimizer import TBPTTOptimizer
        from sandbox.rocky.tf.policies.categorical_rnn_policy import CategoricalRNNPolicy
        from sandbox.rocky.tf.envs.base import TfEnv
        import tensorflow as tf

        from sandbox.rocky.tf.policies.rnn_utils import NetworkType
        from sandbox.rocky.tf.core.network import ConvNetwork
        env = TfEnv(DoomGoalFindingMazeEnv())

        baseline = L2RNNBaseline(
            name="vf",
            env_spec=env.spec,
            feature_network=ConvNetwork(
                name="vf_network",
                input_shape=env.observation_space.shape,
                output_dim=1,
                hidden_sizes=(),
                conv_filters=(16, 32),
                conv_filter_sizes=(5, 5),
                conv_strides=(4, 2),
                conv_pads=('VALID', 'VALID'),
                hidden_nonlinearity=tf.nn.relu,
                output_nonlinearity=tf.nn.relu,
                weight_normalization=v["weight_normalization"],
                batch_normalization=v["batch_normalization"],
            ),
            log_loss_before=False,
            log_loss_after=False,
            hidden_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
            weight_normalization=v["weight_normalization"],
            layer_normalization=v["layer_normalization"],
            state_include_action=False,  # True,
            optimizer=TBPTTOptimizer(
                batch_size=OPT_BATCH_SIZE,
                n_steps=OPT_N_STEPS,
                n_epochs=2,
            ),
            batch_size=OPT_BATCH_SIZE,
            n_steps=OPT_N_STEPS,
        )

        policy = CategoricalRNNPolicy(
            env_spec=env.spec,
            hidden_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
            weight_normalization=v["weight_normalization"],
            layer_normalization=v["layer_normalization"],
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
                weight_normalization=v["weight_normalization"],
                batch_normalization=v["batch_normalization"],
            ),
            name="policy"
        )

        algo = PPOSGD(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=v["batch_size"],
            max_path_length=500,
            max_path_length_schedule= \
                np.concatenate([
                    np.linspace(50, 500, num=v["horizon_schedule_span"], dtype=np.int),
                    np.repeat(500, 500 - v["horizon_schedule_span"])
                ]),
            sampler_args=dict(n_envs=max(1, v["batch_size"] // 500)),
            n_itr=500,
            clip_lr=v["clip_lr"],
            log_loss_kl_before=False,
            log_loss_kl_after=False,
            use_kl_penalty=v["use_kl_penalty"],
            min_n_epochs=2,
            optimizer=TBPTTOptimizer(
                batch_size=OPT_BATCH_SIZE,
                n_steps=OPT_N_STEPS,
                n_epochs=10 if v["use_kl_penalty"] else 2,
            ),
            use_line_search=False
        )
        algo.train()

    config.DOCKER_IMAGE = "dementrock/rllab3-vizdoom-gpu-cuda80"
    config.KUBE_DEFAULT_NODE_SELECTOR = {
        "aws/type": "c4.8xlarge",
    }
    config.KUBE_DEFAULT_RESOURCES = {
        "requests": {
            "cpu": 36 * 0.75,
            "memory": "50Gi",
        },
    }
    env = dict(CUDA_VISIBLE_DEVICES="")

    run_experiment_lite(
        run_task,
        exp_prefix="doom-maze-11-1",
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
