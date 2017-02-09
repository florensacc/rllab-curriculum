import os

from rllab import config

from sandbox.rocky.cirrascale.launch_job import launch_cirrascale

from rllab.misc.instrument import VariantGenerator, variant
from rllab.misc.instrument import run_experiment_lite

"""
Train copter task using TRPO. Fixed to a single task_id
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [1, 2, 3]


USE_GPU = True
# MODE = "local_docker"
MODE = launch_cirrascale("pascal")

vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))


def run_task(v):
    from sandbox.rocky.tf.algos.trpo import TRPO
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.tf.baselines.linear_feature_baseline import LinearFeatureBaseline
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

    env = TfEnv(GprEnv(env_name="I1", seed=0))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(128, 128),
        name="policy",
    )

    from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
    baseline = GaussianMLPBaseline(
        env_spec=env.spec,
        regressor_args=dict(
            use_trust_region=True,
            hidden_sizes=(128, 128),
            optimizer=ConjugateGradientOptimizer(),
            step_size=0.1,
        ),
    )

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=10000,
        max_path_length=100,
        discount=0.99,
        gae_lambda=0.97,
        step_size=0.01,
        n_itr=500,
    )

    algo.train()


if MODE == "local":
    env = dict(PYTHONPATH=":".join([
        config.PROJECT_PATH,
        os.path.join(config.PROJECT_PATH, "conopt_root"),
    ]))
else:
    env = dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/conopt_root")

if MODE in ["local_docker"]:
    env["CUDA_VISIBLE_DEVICES"] = "1"

for vv in variants:
    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="copter-trpo-1",
        mode=MODE,
        n_parallel=8,
        env=env,
        seed=vv["seed"],
        snapshot_mode="last",
        variant=vv,
        terminate_machine=True,
        sync_all_data_node_to_s3=False,
        use_gpu=USE_GPU,
        docker_image="quay.io/openai/rocky-rllab3-conopt-gpu-pascal",
        docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
    )
