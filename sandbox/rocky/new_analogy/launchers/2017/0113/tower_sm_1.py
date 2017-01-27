import os
import pickle

from rllab import config
from rllab.misc import logger

from sandbox.rocky.cirrascale.launch_job import launch_cirrascale

from rllab.misc.instrument import VariantGenerator, variant
from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.s3.resource_manager import resource_manager

"""
TRPO starting near the end state
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]

    @variant
    def n_itr_per_seg(self):
        return [100, 200]

    @variant
    def reward_type(self):
        return ["state", "obs"]


def run_task(v):
    from sandbox.rocky.tf.algos.trpo import TRPO
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    from gpr_package.bin import tower_copter_policy as tower
    from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
    from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
    from sandbox.rocky.new_analogy.envs.traj_reset_env import TrajResetEnv
    from sandbox.rocky.new_analogy.algos.trpo_with_eval_env import TRPOWithEvalEnv
    import tensorflow as tf
    from sandbox.rocky.tf.misc import tensor_utils

    logger.log("Loading data...")
    file_name = resource_manager.get_file("tower_copter_paths_ab_crippled_100")
    with open(file_name, 'rb') as f:
        paths = pickle.load(f)
    logger.log("Loaded")

    task_id = tower.get_task_from_text("ab")

    gpr_env = GprEnv("tower", task_id=task_id, experiment_args=dict(nboxes=2, horizon=1000))

    eval_env = TfEnv(gpr_env)

    policy_params = None

    for k in [1, 5, 10, 20, 40, 80, 160, 320, 640, 1000]:

        tf.reset_default_graph()

        with tf.Session() as sess:

            traj_reset_env = TrajResetEnv(env=gpr_env, paths=paths[:1], k=k, reward_type=v["reward_type"])
            env = TfEnv(traj_reset_env)

            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(128, 128),
                name="policy",
            )

            if policy_params is not None:
                tensor_utils.initialize_new_variables(sess=sess)
                policy.set_param_values(policy_params)

            baseline = GaussianMLPBaseline(
                env_spec=env.spec,
                regressor_args=dict(
                    use_trust_region=True,
                    hidden_sizes=(128, 128),
                    optimizer=ConjugateGradientOptimizer(),
                    step_size=0.1,
                ),
            )

            algo = TRPOWithEvalEnv(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=1000 if k < 100 else 10000,
                max_path_length=k,
                discount=0.995,
                gae_lambda=0.97,
                step_size=0.01,
                n_itr=v["n_itr_per_seg"],
                eval_env=eval_env,
                eval_samples=10000,
                eval_horizon=1000,
                eval_frequency=10,
            )

            algo.train(sess=sess)
            policy_params = policy.get_param_values()


def main():
    USE_GPU = True
    # MODE = "local_docker"
    MODE = launch_cirrascale("pascal")

    vg = VG()

    variants = vg.variants()

    print("#Experiments: %d" % len(variants))

    if MODE == "local":
        env = dict(PYTHONPATH=":".join([
            config.PROJECT_PATH,
            os.path.join(config.PROJECT_PATH, "gpr_package"),
        ]))
    else:
        env = dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package")

    if MODE in ["local_docker"]:
        env["CUDA_VISIBLE_DEVICES"] = "1"

    for vv in variants:
        run_experiment_lite(
            run_task,
            use_cloudpickle=True,
            exp_prefix="tower-sm-1",
            mode=MODE,
            n_parallel=0,
            env=env,
            seed=vv["seed"],
            snapshot_mode="last",
            variant=vv,
            terminate_machine=True,
            sync_all_data_node_to_s3=False,
            use_gpu=USE_GPU,
            docker_image="quay.io/openai/rocky-rllab3-gpr-gpu-pascal:20170112",
            docker_args=" -v /home/rocky/gpr-shared-data:/shared-data",
        )


if __name__ == "__main__":
    main()
