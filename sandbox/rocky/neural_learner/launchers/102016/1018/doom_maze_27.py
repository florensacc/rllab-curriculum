import random
import sys

from rllab.misc.instrument import run_experiment_lite
from rllab import config
from rllab.misc.instrument import VariantGenerator, variant
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
import numpy as np

from sandbox.rocky.tf.core.network import ConvMergeNetwork

"""
Doom multiple episodes - debug
"""

USE_GPU = False#True#False#True
USE_CIRRASCALE = False#True
MODE = "ec2"
# MODE = "local_docker"
# MODE = launch_cirrascale("pascal")
# OPT_BATCH_SIZE = 256
# OPT_N_STEPS = 32#128

N_RNGS = 200#80#38#200


class VG(VariantGenerator):
    entropy_bonus_coeffs = np.exp(np.random.uniform(low=np.log(0.001), high=np.log(0.1), size=N_RNGS))
    discounts = 1. - np.exp(np.random.uniform(low=np.log(0.001), high=np.log(0.01), size=N_RNGS))
    gae_lambdas = 1. - np.exp(np.random.uniform(low=np.log(0.001), high=np.log(0.03), size=N_RNGS))
    mean_kls = np.exp(np.random.uniform(low=np.log(0.001), high=np.log(0.1), size=N_RNGS))
    clip_lrs = np.exp(np.random.uniform(low=np.log(0.1), high=np.log(1.0), size=N_RNGS))
    opt_batch_sizes = np.random.choice(2 ** np.arange(start=1, stop=9), size=N_RNGS)
    min_epochss = np.random.choice([2, 5], size=N_RNGS)
    hidden_dims = np.random.choice([32, 128], size=N_RNGS)
    horizon_schedule_spans = np.random.choice([0], size=N_RNGS)
    batch_sizes = np.random.choice([10000, 25000, 50000], size=N_RNGS)
    nonlinearities = np.random.choice(["relu", "tanh"], size=N_RNGS)
    n_episodess = np.random.choice([5, 10], size=N_RNGS)
    episode_horizons = np.random.choice([100, 300, 500], size=N_RNGS)

    @variant
    def seed(self, rng_index):
        yield int(rng_index * 10 + 1 + 42345)

    @variant
    def rng_index(self):
        return list(range(N_RNGS))

    @variant
    def horizon_schedule_span(self, rng_index):
        yield int(self.horizon_schedule_spans[rng_index])

    @variant
    def batch_size(self, rng_index):
        if MODE == "local":
            yield 10000
        else:
            yield int(self.batch_sizes[rng_index])

    @variant
    def docker_image(self):
        return [
            "dementrock/rllab3-vizdoom-gpu-cuda80:cig",
        ]

    @variant
    def rescale_obs(self):
        return [(30, 40)]

    @variant
    def clip_lr(self, rng_index):
        yield float(self.clip_lrs[rng_index])

    @variant
    def use_kl_penalty(self):
        return [False]

    @variant
    def nonlinearity(self, rng_index):
        yield str(self.nonlinearities[rng_index])

    @variant
    def layer_normalization(self):
        return [False]

    @variant
    def weight_normalization(self):
        return [True]

    @variant
    def mean_kl(self, rng_index):
        yield float(self.mean_kls[rng_index])

    @variant
    def n_episodes(self, rng_index):
        yield int(self.n_episodess[rng_index])

    @variant
    def episode_horizon(self, rng_index):
        yield int(self.episode_horizons[rng_index])

    @variant
    def min_epochs(self, rng_index):
        yield int(self.min_epochss[rng_index])

    @variant
    def opt_batch_size(self, rng_index):
        yield int(self.opt_batch_sizes[rng_index])

    @variant
    def opt_n_steps(self):
        return [None]

    @variant
    def batch_normalization(self):
        return [False]

    @variant
    def entropy_bonus_coeff(self, rng_index):
        yield float(self.entropy_bonus_coeffs[rng_index])

    @variant
    def discount(self, rng_index):
        yield float(self.discounts[rng_index])

    @variant
    def gae_lambda(self, rng_index):
        yield float(self.gae_lambdas[rng_index])

    @variant
    def hidden_dim(self, rng_index):
        yield int(self.hidden_dims[rng_index])

    @variant
    def n_itr(self):
        yield 1000


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for vv in variants:  # [:2]:

    def run_task(v):
        from sandbox.rocky.neural_learner.baselines.l2_rnn_baseline import L2RNNBaseline
        from sandbox.rocky.neural_learner.envs.doom_goal_finding_maze_env import DoomGoalFindingMazeEnv
        from sandbox.rocky.neural_learner.algos.pposgd_clip_ratio import PPOSGD
        from sandbox.rocky.neural_learner.optimizers.tbptt_optimizer import TBPTTOptimizer
        from sandbox.rocky.neural_learner.policies.categorical_rnn_policy import CategoricalRNNPolicy
        from sandbox.rocky.tf.envs.base import TfEnv
        from sandbox.rocky.neural_learner.envs.doom_hex_goal_finding_maze_env import DoomHexGoalFindingMazeEnv
        from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv
        import tensorflow as tf

        from sandbox.rocky.tf.policies.rnn_utils import NetworkType
        from sandbox.rocky.tf.core.network import ConvNetwork
        env = TfEnv(
            MultiEnv(
                wrapped_env=DoomHexGoalFindingMazeEnv(
                    rescale_obs=v["rescale_obs"],
                    reset_map=False,
                    randomize_texture=False,
                    difficulty=1,
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
            state_include_action=False,  # True,
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
            state_include_action=False,
            feature_network=new_feature_network("embedding_network", output_dim=256),
            name="policy"
        )

        max_path_length = v["n_episodes"] * v["episode_horizon"]

        algo = PPOSGD(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=v["batch_size"],
            max_path_length=max_path_length,
            max_path_length_schedule= \
                np.concatenate([
                    np.linspace(50, max_path_length, num=v["horizon_schedule_span"], dtype=np.int),
                    np.repeat(max_path_length, v["n_itr"] - v["horizon_schedule_span"])
                ]),
            sampler_args=dict(n_envs=max(1, v["batch_size"] // max_path_length)),
            n_itr=v["n_itr"],
            clip_lr=v["clip_lr"],
            log_loss_kl_before=False,
            log_loss_kl_after=False,
            use_kl_penalty=v["use_kl_penalty"],
            min_n_epochs=v["min_epochs"],
            discount=v["discount"],
            gae_lambda=v["gae_lambda"],
            optimizer=TBPTTOptimizer(
                batch_size=v["opt_batch_size"],
                n_steps=v["opt_n_steps"],
                n_epochs=v["min_epochs"],
            ),
            entropy_bonus_coeff=v["entropy_bonus_coeff"],
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
    config.AWS_REGION_NAME = random.choice(['us-west-1'])#, 'us-west-1', 'us-east-1'])
    config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]
    config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[config.AWS_REGION_NAME]
    config.AWS_SECURITY_GROUP_IDS = config.ALL_REGION_AWS_SECURITY_GROUP_IDS[config.AWS_REGION_NAME]

    if MODE == "local_docker":
        env = dict(CUDA_VISIBLE_DEVICES="3")
    else:
        env = dict()  # CUDA_VISIBLE_DEVICES="")

    run_experiment_lite(
        run_task,
        exp_prefix="doom-maze-27-1",
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
