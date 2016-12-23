# First experiment with claw, using behavior cloning
# The end goal is to get some reasonable behavior using as few demonstrations as possible
from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.new_analogy.policies.normalizing_policy import NormalizingPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc import logger
import numpy as np


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]

    @variant
    def init_std(self):
        return [0.3]

    @variant
    def disc_learning_rate(self):
        return [1e-4]

    @variant
    def disc_include_actions(self):
        return [False]

    @variant
    def n_trajs(self):
        return ["20k"]

    @variant
    def demo_mixture_ratio(self):
        return [0., 0.1, 0.3, 0.5, 0.7][::-1]


def run_task(vv):
    import tensorflow as tf
    from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
    from sandbox.rocky.tf.algos.trpo import TRPO
    from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
    from sandbox.rocky.new_analogy.discriminators.mlp_discriminator import MLPDiscriminator
    from sandbox.rocky.new_analogy.envs.conopt_env import ConoptEnv
    from sandbox.rocky.new_analogy.sample_processors.gail_sample_processor import GAILSampleProcessor
    from sandbox.rocky.s3.resource_manager import resource_manager
    import joblib

    with tf.Session() as sess:
        resource_name = "irl/claw-bc-pretrained-v1.pkl"

        logger.log("Loading data...")
        data = np.load("/shared-data/claw-{n_trajs}-data.npz".format(n_trajs=vv["n_trajs"]))
        exp_x = data["exp_x"]
        exp_u = data["exp_u"]
        exp_rewards = data["exp_rewards"]
        logger.log("Loaded")
        paths = []

        for xs, us, rewards in zip(exp_x, exp_u, exp_rewards):
            if rewards[-1] > 4.5:
                paths.append(dict(observations=xs, actions=us, rewards=rewards))

        file_name = resource_manager.get_file(resource_name=resource_name)

        policy = joblib.load(file_name)["policy"]

        env = TfEnv(ConoptEnv("TF2"))

        discriminator = MLPDiscriminator(
            env_spec=env.spec,
            demo_paths=paths,
            n_epochs=2,
            discount=0.995,
            hidden_sizes=(256, 256, 256),
            hidden_nonlinearity=tf.nn.relu,
            zero_expert_discr=False,
            sink_zero_expert_discr=False,
            use_sink_rewards=True,
            learning_rate=v["disc_learning_rate"],
            demo_mixture_ratio=v["demo_mixture_ratio"],
            include_actions=v["disc_include_actions"],
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                use_trust_region=True,
                hidden_sizes=(256, 256, 256),
                hidden_nonlinearity=tf.nn.relu,
                optimizer=ConjugateGradientOptimizer(),
                step_size=0.1,
            ),
        )

        sess.run(tf.assign(policy._l_std_param.param, [np.log(vv["init_std"])] * env.action_dim))

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=50000,
            max_path_length=100,
            n_itr=5000,
            discount=0.995,
            gae_lambda=0.97,
            parallel_vec_env=True,
            n_vectorized_envs=100,
            sample_processor_cls=GAILSampleProcessor,
            sample_processor_args=dict(discriminator=discriminator),
        )

        algo.train(sess=sess)


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:
    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="gail-finetune-claw-2",
        # mode="local_docker",
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
