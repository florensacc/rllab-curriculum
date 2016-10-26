from rllab.misc.instrument import run_experiment_lite
from rllab import config
from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]

    @variant
    def batch_size(self):
        return [10000, 50000]


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:


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
                batch_normalization=False,
            ),
            hidden_nonlinearity=tf.nn.relu,
            weight_normalization=True,
            layer_normalization=False,
            state_include_action=False,  # True,
            optimizer=TBPTTOptimizer(
                batch_size=256,
                n_steps=20,
                n_epochs=3,
            ),
        )

        policy = CategoricalRNNPolicy(
            env_spec=env.spec,
            hidden_nonlinearity=tf.nn.relu,
            weight_normalization=True,
            layer_normalization=False,
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
                batch_normalization=False,
            ),
            name="policy"
        )

        algo = PPOSGD(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=variant["batch_size"],
            max_path_length=500,
            sampler_args=dict(n_envs=20),
            n_itr=1000,
            clip_lr=0.2,
            log_loss_kl_before=False,
            log_loss_kl_after=False,
            optimizer=TBPTTOptimizer(
                batch_size=256,
                n_steps=20,
                n_epochs=3,
            )
            # n_epochs=0,
        )
        algo.train()


    USE_GPU = False  # True#False

    if USE_GPU:
        config.DOCKER_IMAGE = "dementrock/rllab3-vizdoom-gpu-cuda80"
    else:
        config.DOCKER_IMAGE = "dementrock/rllab3-vizdoom-gpu-cuda80"

    config.KUBE_DEFAULT_NODE_SELECTOR = {
        "aws/type": "g2.2xlarge",
    }
    config.KUBE_DEFAULT_RESOURCES = {
        "requests": {
            "cpu": 4 * 0.75,
        },
    }

    run_experiment_lite(
        run_task,
        exp_prefix="doom_maze_2",
        mode="lab_kube",
        n_parallel=0,
        seed=v["seed"],
        use_gpu=USE_GPU,
        use_cloudpickle=True,
        variant=v,
        snapshot_mode="last",
    )
