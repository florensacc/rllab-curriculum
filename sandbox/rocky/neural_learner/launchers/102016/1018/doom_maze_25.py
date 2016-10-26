from rllab.misc.instrument import run_experiment_lite
from rllab import config
from rllab.misc.instrument import VariantGenerator, variant
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
import numpy as np

from sandbox.rocky.neural_learner.policies.categorical_rnn_policy import SoftmaxDefault, SoftmaxExactEntropy
from sandbox.rocky.neural_learner.policies.categorical_rnn_policy import SoftmaxNormalized

"""
Train Doom on the new hexagonal task
"""

USE_GPU = False#True
USE_CIRRASCALE = True
MODE = "ec2"
# MODE = launch_cirrascale("pascal")
# OPT_BATCH_SIZE = 256
# OPT_N_STEPS = 32#128


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]

    @variant
    def seed_copy(self, seed):
        yield seed

    @variant
    def horizon_schedule_span(self):
        return [0]#, 100, 500]

    @variant
    def batch_size(self):
        if MODE == "local":
            return [10000]
        return [50000]#10000]#, 50000]

    @variant
    def docker_image(self):
        return [
            # "dementrock/rllab3-vizdoom-gpu-cuda80:master",
            "dementrock/rllab3-vizdoom-gpu-cuda80:cig",
        ]

    @variant
    def rescale_obs(self):
        return [(60, 80), (30, 40)]

    @variant
    def clip_lr(self):
        return [0.3]

    @variant
    def use_kl_penalty(self):
        return [False]#True]

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
        return ["default"]

    @variant
    def min_epochs(self):
        return [5, 10]#, 20]

    @variant
    def opt_batch_size(self):
        return [32]

    @variant
    def opt_n_steps(self):
        return [None]#32, 128, None]

    @variant
    def batch_normalization(self):
        return [False]

    @variant
    def difficulty(self):
        return [-5, -3, 1, 3, 5]

vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for vv in variants:  # [:2]:

    def run_task(v):
        from sandbox.rocky.neural_learner.baselines.l2_rnn_baseline import L2RNNBaseline
        from sandbox.rocky.neural_learner.envs.doom_hex_goal_finding_maze_env import DoomHexGoalFindingMazeEnv
        from sandbox.rocky.neural_learner.algos.pposgd_clip_ratio import PPOSGD
        from sandbox.rocky.neural_learner.optimizers.tbptt_optimizer import TBPTTOptimizer
        from sandbox.rocky.neural_learner.policies.categorical_rnn_policy import CategoricalRNNPolicy, \
            SoftmaxExactEntropy
        from sandbox.rocky.tf.envs.base import TfEnv
        import tensorflow as tf

        from sandbox.rocky.tf.policies.rnn_utils import NetworkType
        from sandbox.rocky.tf.core.network import ConvNetwork
        env = TfEnv(
            DoomHexGoalFindingMazeEnv(
                rescale_obs=v["rescale_obs"],
                difficulty=v["difficulty"]
            )
        )

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
                batch_size=v["opt_batch_size"],
                n_steps=v["opt_n_steps"],
                n_epochs=v["min_epochs"],
            ),
            batch_size=v["opt_batch_size"],
            n_steps=v["opt_n_steps"],
        )

        if v["softmax_param"] == "default":
            action_param = SoftmaxDefault(dim=4)
        elif v["softmax_param"] == "normalized":
            action_param = SoftmaxNormalized(dim=4, bias=1.)
        elif v["softmax_param"] == "exact_entropy":
            action_param = SoftmaxExactEntropy(
                dim=4,
                input_dependent=False,
                initial_entropy_percentage=v["initial_entropy_percentage"]
            )
        elif v["softmax_param"] == "exact_entropy_dependent":
            action_param = SoftmaxExactEntropy(dim=4, input_dependent=True)
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
                batch_size=v["opt_batch_size"],
                n_steps=v["opt_n_steps"],
                n_epochs=v["min_epochs"],
            ),
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
    config.AWS_REGION_NAME = 'us-west-2'
    config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]
    config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[config.AWS_REGION_NAME]
    config.AWS_SECURITY_GROUP_IDS = config.ALL_REGION_AWS_SECURITY_GROUP_IDS[config.AWS_REGION_NAME]

    if MODE == "local_docker":
        env = dict(CUDA_VISIBLE_DEVICES="3")
    else:
        env = dict()  # CUDA_VISIBLE_DEVICES="")

    run_experiment_lite(
        run_task,
        exp_prefix="doom-maze-25-ec2-1",
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
