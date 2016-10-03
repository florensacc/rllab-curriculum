from rllab.misc.instrument import run_experiment_lite
from rllab import config
from rllab.misc.instrument import VariantGenerator, variant

USE_GPU = False#True#False  # True#False
USE_CIRRASCALE = True
MODE = "lab_kube"
OPT_BATCH_SIZE = 256
OPT_N_STEPS = 32

class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11]#, 21, 31]

    @variant
    def batch_size(self):
        if MODE == "local":
            return [2000]
        return [10000]

    @variant
    def clip_lr(self):
        return [0.1]#, 0.2]

    @variant
    def use_kl_penalty(self):
        return [True, False]#0.1, 0.2, 0.05]

    @variant
    def nonlinearity(self):
        return ["relu", "tanh"]

    @variant
    def layer_normalization(self):
        return [True, False]

    @variant
    def batch_normalization(self):
        return [True, False]


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:#[:10]:

    def run_task(variant):
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
                weight_normalization=True,
                batch_normalization=v["batch_normalization"],
            ),
            log_loss_before=False,
            log_loss_after=False,
            hidden_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
            weight_normalization=True,
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
            weight_normalization=True,
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
                weight_normalization=True,
                batch_normalization=v["batch_normalization"],
            ),
            name="policy"
        )

        algo = PPOSGD(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=variant["batch_size"],
            max_path_length=500,
            sampler_args=dict(n_envs=max(1, variant["batch_size"] // 500)),
            n_itr=1000,
            clip_lr=v["clip_lr"],
            log_loss_kl_before=False,
            log_loss_kl_after=False,
            use_kl_penalty=v["use_kl_penalty"],
            min_n_epochs=2,
            optimizer=TBPTTOptimizer(
                batch_size=OPT_BATCH_SIZE,
                n_steps=OPT_N_STEPS,
                n_epochs=10,
            )
        )
        algo.train()



    if USE_GPU:
        if USE_CIRRASCALE:
            config.DOCKER_IMAGE = "dementrock/rllab3-vizdoom-gpu-cuda80"
            config.KUBE_DEFAULT_NODE_SELECTOR = {
                "openai.org/machine-class": "cirrascale",
                "openai.org/gpu-type": "titanx-pascal",
            }
            if MODE == "local_docker":
                env = dict(OPENAI_N_GPUS="1", CUDA_VISIBLE_DEVICES="1")
            else:
                env = dict(OPENAI_N_GPUS="1")
        else:
            if MODE == "local_docker":
                config.DOCKER_IMAGE = "dementrock/rllab3-vizdoom-gpu-cuda80"
            else:
                config.DOCKER_IMAGE = "dementrock/rllab3-vizdoom-gpu"
            config.KUBE_DEFAULT_NODE_SELECTOR = {
                "aws/type": "g2.2xlarge",
            }
            config.KUBE_DEFAULT_RESOURCES = {
                "requests": {
                    "cpu": 4 * 0.75,
                },
                "limits": {
                    "cpu": 4 * 0.75,
                },
            }
            env = dict(CUDA_VISIBLE_DEVICES="1")
    else:
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
        exp_prefix="doom_maze_3_cpu_1",
        mode=MODE,
        n_parallel=0,
        seed=v["seed"],
        use_gpu=USE_GPU,
        use_cloudpickle=True,
        variant=v,
        snapshot_mode="last",
        env=env,
        terminate_machine=True,
        sync_all_data_node_to_s3=False,
    )
    # sys.exit()
