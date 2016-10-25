from rllab.misc.instrument import run_experiment_lite
from rllab import config
from rllab.misc.instrument import VariantGenerator, variant
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
import numpy as np

from sandbox.rocky.neural_learner.policies.categorical_rnn_policy import SoftmaxDefault, SoftmaxExactEntropy
from sandbox.rocky.neural_learner.policies.categorical_rnn_policy import SoftmaxNormalized

"""
Test exact entropy
"""

USE_GPU = True
USE_CIRRASCALE = True
# MODE = "local_docker"
MODE = launch_cirrascale#()
OPT_BATCH_SIZE = 256
OPT_N_STEPS = 32#128


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31, 41, 51]

    @variant
    def seed_copy(self, seed):
        yield seed

    @variant
    def horizon_schedule_span(self):
        return [100]

    @variant
    def batch_size(self):
        if MODE == "local":
            return [10000]
        return [10000, 50000]

    @variant
    def docker_image(self):
        return [
            "dementrock/rllab3-vizdoom-gpu-cuda80:cig",
        ]

    @variant
    def rescale_obs(self):
        return [(120, 160)]

    @variant
    def clip_lr(self):
        return [0.3]

    @variant
    def use_kl_penalty(self):
        return [True]

    @variant
    def nonlinearity(self):
        return ["relu"]

    @variant
    def layer_normalization(self):
        return [False]

    @variant
    def weight_normalization(self):
        return [True]

    @variant
    def softmax_param(self):
        return ["exact_entropy", "default", "normalized"]

    @variant
    def min_epochs(self):
        return [10]

    @variant
    def vf_min_epochs(self):
        return [2]

    @variant
    def batch_normalization(self):
        return [False]


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for vv in variants:  # [:2]:

    def run_task(v):
        from sandbox.rocky.neural_learner.baselines.l2_rnn_baseline import L2RNNBaseline
        from sandbox.rocky.neural_learner.envs.doom_goal_finding_maze_env import DoomGoalFindingMazeEnv
        from sandbox.rocky.neural_learner.algos.pposgd_clip_ratio import PPOSGD
        from sandbox.rocky.neural_learner.optimizers.tbptt_optimizer import TBPTTOptimizer
        from sandbox.rocky.neural_learner.policies.categorical_rnn_policy import CategoricalRNNPolicy, \
            SoftmaxExactEntropy
        from sandbox.rocky.tf.envs.base import TfEnv
        import tensorflow as tf

        from sandbox.rocky.tf.policies.rnn_utils import NetworkType
        from sandbox.rocky.tf.core.network import ConvNetwork
        env = TfEnv(DoomGoalFindingMazeEnv(rescale_obs=v["rescale_obs"]))

        def new_feature_network(name, output_dim):
            return ConvNetwork(
                name=name,
                input_shape=env.observation_space.shape,
                output_dim=output_dim,
                hidden_sizes=(),
                conv_filters=(16, 32) if v["rescale_obs"] == (120, 160) else (16, 16),
                conv_filter_sizes=(5, 5),
                conv_strides=(4, 2) if v["rescale_obs"] == (120, 160) else (2, 2),
                conv_pads=('VALID', 'VALID'),
                hidden_nonlinearity=tf.nn.relu,
                output_nonlinearity=tf.nn.relu,
                weight_normalization=v["weight_normalization"],
                batch_normalization=v["batch_normalization"],
            )

        baseline = L2RNNBaseline(
            name="vf",
            env_spec=env.spec,
            feature_network=new_feature_network("vf_network", 1),
            log_loss_before=False,
            log_loss_after=False,
            hidden_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
            weight_normalization=v["weight_normalization"],
            layer_normalization=v["layer_normalization"],
            state_include_action=False,  # True,
            optimizer=TBPTTOptimizer(
                batch_size=OPT_BATCH_SIZE,
                n_steps=OPT_N_STEPS,
                n_epochs=v["vf_min_epochs"],
            ),
            batch_size=OPT_BATCH_SIZE,
            n_steps=OPT_N_STEPS,
        )

        if v["softmax_param"] == "default":
            action_param = SoftmaxDefault(dim=4)
        elif v["softmax_param"] == "normalized":
            action_param = SoftmaxNormalized(dim=4, bias=1.)
        elif v["softmax_param"] == "exact_entropy":
            action_param = SoftmaxExactEntropy(dim=4)
        else:
            raise NotImplementedError

        policy = CategoricalRNNPolicy(
            env_spec=env.spec,
            hidden_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
            weight_normalization=v["weight_normalization"],
            layer_normalization=v["layer_normalization"],
            network_type=NetworkType.GRU,
            state_include_action=False,
            action_param=action_param,
            feature_network=new_feature_network("embedding_network", output_dim=256),
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
            min_n_epochs=v["min_epochs"],
            optimizer=TBPTTOptimizer(
                batch_size=OPT_BATCH_SIZE,
                n_steps=OPT_N_STEPS,
                n_epochs=10,
            ),
            use_line_search=False#True
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
    if MODE == "local_docker":
        env = dict(CUDA_VISIBLE_DEVICES="1")
    else:
        env = dict()  # CUDA_VISIBLE_DEVICES="")

    run_experiment_lite(
        run_task,
        exp_prefix="doom-maze-17-4",
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
        # python_command="kernprof -o /root/code/rllab/doom.lprof -l",
    )
    # sys.exit()
