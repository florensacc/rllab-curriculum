# First experiment with claw, using behavior cloning
# The end goal is to get some reasonable behavior using as few demonstrations as possible
import numpy as np

from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.new_analogy.tf.policies.normalizing_policy import NormalizingPolicy
from sandbox.rocky.tf.envs.base import TfEnv


class VG(VariantGenerator):
    @variant
    def algo(self):
        return [
            "trpo",
            # "gail-trpo",
        ]

    @variant
    def seed(self):
        return [11, 21, 31, 41, 51]

    @variant
    def disc_learning_rate(self):
        return [1e-4]

    @variant
    def normalize_policy(self):
        return [True]


def run_task(vv):
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.tf.algos.trpo import TRPO
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    import tensorflow as tf
    from sandbox.rocky.new_analogy.tf.discriminators import MLPDiscriminator
    from sandbox.rocky.new_analogy.sample_processors.gail_sample_processor import GAILSampleProcessor
    from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline

    from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

    logger.log("Loading data...")
    data = np.load("/shared-data/claw-2k-data.npz")
    exp_x = data["exp_x"]
    exp_u = data["exp_u"]
    exp_rewards = data["exp_rewards"]
    logger.log("Loaded")
    paths = []

    for xs, us, rewards in zip(exp_x, exp_u, exp_rewards):
        if rewards[-1] > 4.5:
            paths.append(dict(observations=xs, actions=us, rewards=rewards))

    env = TfEnv(GprEnv("TF2"))

    discriminator = MLPDiscriminator(
        env_spec=env.spec,
        demo_paths=paths,
        n_epochs=2,
        discount=0.99,
        hidden_sizes=(128, 128),
        zero_expert_discr=False,
        sink_zero_expert_discr=False,
        use_sink_rewards=True,
        learning_rate=v["disc_learning_rate"],  # 1e-4,
    )

    baseline = GaussianMLPBaseline(
        env_spec=env.spec,
        regressor_args=dict(
            use_trust_region=True,
            hidden_sizes=(128, 128),
            optimizer=ConjugateGradientOptimizer(),
            step_size=0.1,
        ),
    )

    policy = NormalizingPolicy(
        GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(128, 128),
            hidden_nonlinearity=tf.nn.relu,
            name="policy"
        ),
        paths=paths,
        normalize_actions=v["normalize_policy"],
        normalize_obs=v["normalize_policy"],
    )

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=50000,
        max_path_length=100,
        n_itr=5000,
        discount=0.995,
        gae_lambda=0.97,
        sample_processor_cls=GAILSampleProcessor if "gail" in vv["algo"] else None,
        sample_processor_args=dict(discriminator=discriminator) if "gail" in vv["algo"] else None,
        parallel_vec_env=True,
        n_vectorized_envs=100,
    )

    algo.train()


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:
    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="gail-claw-5",
        mode=launch_cirrascale("pascal"),
        use_gpu=True,
        snapshot_mode="last",
        sync_all_data_node_to_s3=False,
        n_parallel=8,
        env=dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/conopt_root"),
        docker_image="quay.io/openai/rocky-rllab3-conopt-gpu-pascal",
        docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
        variant=v,
        seed=v["seed"],
    )
