from rllab.misc.instrument import run_experiment_lite
from rllab import config
from rllab.misc.instrument import VariantGenerator, variant
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
import numpy as np

from sandbox.rocky.neural_learner.algos.trpo import TRPO
from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv
from sandbox.rocky.neural_learner.policies.categorical_rnn_policy import SoftmaxDefault, SoftmaxExactEntropy
from sandbox.rocky.neural_learner.policies.categorical_rnn_policy import SoftmaxNormalized
from sandbox.rocky.tf.core.network import ConvMergeNetwork
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

"""
Make the target larger again...
"""

USE_GPU = True
USE_CIRRASCALE = True

# MODE = "local"
# MODE = "local_docker"
MODE = launch_cirrascale("pascal")


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [10 * x + 1 for x in range(5)]

    @variant
    def batch_size(self):
        if MODE == "local":
            return [10000]
        return [50000]#20000, 100000]

    @variant
    def docker_image(self):
        return [
            "dementrock/rllab3-vizdoom-gpu-cuda80:cig",
        ]

    @variant
    def rescale_obs(self):
        return [(30, 40)]

    @variant
    def clip_lr(self):
        return [0.1]#, 0.1]

    @variant
    def use_kl_penalty(self):
        return [False]

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
        return [5]

    @variant
    def opt_batch_size(self):
        return [32]

    @variant
    def opt_n_steps(self):
        return [None]

    @variant
    def batch_normalization(self):
        return [False]

    @variant
    def n_episodes(self):
        return [2]#10, 5, 2]

    @variant
    def episode_horizon(self, n_episodes):
        yield 250#500

    @variant
    def max_path_length(self, n_episodes, episode_horizon):
        yield n_episodes * episode_horizon

    @variant
    def difficulty(self):
        return [1]

    # @variant
    # def rand_start(self):
    #     return [True, False]

    @variant
    def n_repeats(self):
        return [1]

    @variant
    def n_targets(self):
        return [1]

    @variant
    def randomize_texture(self):
        return [False]

    @variant
    def discount(self):
        return [0.99]

    @variant
    def gae_lambda(self):
        return [0.99]#, 0.7]

    @variant
    def hidden_dim(self):
        return [256]

    @variant
    def living_reward(self):
        return [-0.01]#, -0.1, -1.]

    @variant
    def frame_skip(self):
        return [4]

    @variant
    def n_itr(self):
        return [1000]

    @variant
    def margin(self):
        return [1]

    @variant
    def side_length(self):
        return [96]

    @variant
    def wall_penalty(self):
        return [0., 0.001, 0.0001, 0.01]

vg = VG()

variants = vg.variants(randomized=True)

print("#Experiments: %d" % len(variants))

for idx, vv in enumerate(variants):

    def run_task(v):
        from sandbox.rocky.neural_learner.baselines.l2_rnn_baseline import L2RNNBaseline
        from sandbox.rocky.neural_learner.algos.pposgd_clip_ratio import PPOSGD
        from sandbox.rocky.neural_learner.optimizers.tbptt_optimizer import TBPTTOptimizer
        from sandbox.rocky.neural_learner.policies.categorical_rnn_policy import CategoricalRNNPolicy
        from sandbox.rocky.tf.envs.base import TfEnv
        import tensorflow as tf
        from sandbox.rocky.neural_learner.envs.doom_fixed_goal_finding_maze_env import DoomFixedGoalFindingMazeEnv
        from sandbox.rocky.neural_learner.envs.doom_goal_finding_maze_env import DoomGoalFindingMazeEnv
        from sandbox.rocky.tf.policies.rnn_utils import NetworkType

        from sandbox.rocky.neural_learner.envs.doom_po_env import DoomPoEnv
        env = TfEnv(
            MultiEnv(
                wrapped_env=DoomFixedGoalFindingMazeEnv(
                    rescale_obs=v["rescale_obs"],
                    reset_map=False,
                    living_reward=v["living_reward"],
                    frame_skip=v["frame_skip"],
                    allow_backwards=False,
                    margin=v["margin"],
                    side_length=v["side_length"],
                    n_trajs=2,
                    # maze_sizes=[3],
                    version="v1",
                    map_preset="two_goal_lv2",
                    wall_penalty=v["wall_penalty"],
                ),
                n_episodes=v["n_episodes"],
                episode_horizon=v["episode_horizon"],
                discount=v["discount"],
            )
        )

        def new_feature_network(name, output_dim):
            img_space = env.observation_space.components[0]
            img_shape = img_space.shape
            extra_dim = int(env.observation_space.flat_dim - img_space.flat_dim)
            return ConvMergeNetwork(
                name=name,
                input_shape=img_shape,
                extra_input_shape=(extra_dim,),
                output_dim=output_dim,
                hidden_sizes=(v["hidden_dim"],),
                extra_hidden_sizes=(v["hidden_dim"],),
                conv_filters=(16, 32) if v["rescale_obs"] == (120, 160) else (16, 16),
                conv_filter_sizes=(5, 5),
                conv_strides=(4, 2) if v["rescale_obs"] == (120, 160) else (2, 2),
                conv_pads=('VALID', 'VALID'),
                hidden_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
                output_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
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
            state_include_action=False,
            feature_network=new_feature_network("embedding_network", output_dim=v["hidden_dim"]),
            name="policy"
        )

        if MODE == "ec2":
            MAX_ENVS = 36
        elif MODE == "local_docker":
            MAX_ENVS = 256
        elif MODE == "local":
            MAX_ENVS = 4
        elif "launch" in str(MODE):
            MAX_ENVS = 256
        else:
            raise NotImplementedError

        n_envs = min(MAX_ENVS, max(1, v["batch_size"] // v["max_path_length"]))

        algo = PPOSGD(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=v["batch_size"],
            max_path_length=v["max_path_length"],
            n_vectorized_envs=n_envs,
            n_itr=v["n_itr"],
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
            use_line_search=True,
            discount=v["discount"],
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
    config.AWS_INSTANCE_TYPE = ["m4.16xlarge", "m4.10xlarge"][idx % 2]
    config.AWS_SPOT = True
    config.AWS_SPOT_PRICE = '1.0'
    config.AWS_REGION_NAME = ['us-west-2', 'us-west-1', 'us-east-1'][idx % 3]
    config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]
    config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[config.AWS_REGION_NAME]
    config.AWS_SECURITY_GROUP_IDS = config.ALL_REGION_AWS_SECURITY_GROUP_IDS[config.AWS_REGION_NAME]

    if MODE == "local_docker":
        env = dict(CUDA_VISIBLE_DEVICES="3")
    else:
        env = dict()

    run_experiment_lite(
        run_task,
        exp_prefix="doom-maze-71",
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
