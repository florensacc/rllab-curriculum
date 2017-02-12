# First experiment with claw, using behavior cloning
# The end goal is to get some reasonable behavior using as few demonstrations as possible
from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.tf.envs.base import TfEnv


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]

    @variant
    def disc_learning_rate(self):
        return [1e-4]

    @variant
    def disc_include_actions(self):
        return [True, False]

    @variant
    def disc_cost_form(self):
        return ["linear", "sum_square", "square", "softplus"]

    @variant
    def demo_mixture_ratio(self):
        return [0.5]#, 0.1, 0.3, 0.5, 0.7][::-1]


def run_task(vv):
    import tensorflow as tf
    from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
    from sandbox.rocky.tf.algos.trpo import TRPO
    from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.new_analogy.tf.discriminators import GCLCostLearner
    from sandbox.rocky.new_analogy.sample_processors.gail_sample_processor import GAILSampleProcessor
    from sandbox.rocky.s3.resource_manager import resource_manager
    from rllab.envs.normalized_env import normalize
    from rllab.envs.mujoco.swimmer_env import SwimmerEnv
    import cloudpickle

    task = "swimmer"
    n_trajs = 1000
    horizon = 500

    deterministic = False  # True

    traj_resource_name = "demo_trajs/{task}_n_trajs_{n_trajs}_horizon_{horizon}_deterministic_{" \
                         "deterministic}.pkl".format(task=task, n_trajs=str(n_trajs), horizon=str(
        horizon), deterministic=str(deterministic))

    print(traj_resource_name)

    with open(resource_manager.get_file(traj_resource_name), "rb") as f:
        demo_paths = cloudpickle.load(f)

    env = TfEnv(normalize(SwimmerEnv()))

    policy = GaussianMLPPolicy(name="policy", env_spec=env.spec, hidden_sizes=(100, 100))

    discriminator = GCLCostLearner(
        env_spec=env.spec,
        demo_paths=demo_paths,
        policy=policy,
        n_epochs=2,
        hidden_sizes=(256, 256),
        hidden_nonlinearity=tf.nn.relu,
        learning_rate=v["disc_learning_rate"],
        demo_mixture_ratio=v["demo_mixture_ratio"],
        include_actions=v["disc_include_actions"],
        cost_form=v["disc_cost_form"],
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

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=50000,
        max_path_length=500,
        n_itr=5000,
        discount=0.995,
        gae_lambda=0.97,
        parallel_vec_env=True,
        n_vectorized_envs=100,
        sample_processor_cls=GAILSampleProcessor,
        sample_processor_args=dict(discriminator=discriminator),
    )

    algo.train()


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:
    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="gcl-1-1",
        # mode="local_docker",
        mode=launch_cirrascale("pascal"),
        use_gpu=True,
        snapshot_mode="last",
        sync_all_data_node_to_s3=False,
        n_parallel=8,
        # env=dict(CUDA_VISIBLE_DEVICES="1"),#PYTHONPATH="/root/code/rllab:/root/code/rllab/conopt_root"),
        docker_image="quay.io/openai/rocky-rllab3-conopt-gpu-pascal",
        # docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
        variant=v,
        seed=v["seed"],
    )
    # break
