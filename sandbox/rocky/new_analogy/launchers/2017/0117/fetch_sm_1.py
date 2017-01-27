import os
import pickle
import random

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


def run_task(v):
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
    from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
    from sandbox.rocky.new_analogy.algos.trpo_with_eval_env import TRPOWithEvalEnv
    from sandbox.rocky.new_analogy.envs.fetch_reduced import FetchReduced
    import tensorflow as tf
    from sandbox.rocky.tf.misc import tensor_utils

    logger.log("Loading data...")
    file_name = resource_manager.get_file("fetch_single_traj")
    with open(file_name, 'rb') as f:
        ref_path = pickle.load(f)
    logger.log("Loaded")

    eval_env = TfEnv(FetchReduced(ref_path=ref_path, k=1000, reward_type='original'))

    policy_params = None

    for k in [1, 5, 10, 20, 40, 80, 160, 320, 640, 1000]:

        tf.reset_default_graph()

        with tf.Session() as sess:

            env = TfEnv(FetchReduced(ref_path=ref_path, k=k, reward_type='shaped'))

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
    USE_GPU = False  # True
    # MODE = "local_docker"
    # MODE = launch_cirrascale("pascal")
    MODE = "ec2"

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
        config.AWS_INSTANCE_TYPE = "c4.8xlarge"
        config.AWS_SPOT = True
        config.AWS_SPOT_PRICE = '2.0'
        config.AWS_REGION_NAME = random.choice(
            ['us-west-1', 'us-east-1', 'us-west-2']
        )
        config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]
        config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[config.AWS_REGION_NAME]
        config.AWS_SECURITY_GROUP_IDS = config.ALL_REGION_AWS_SECURITY_GROUP_IDS[config.AWS_REGION_NAME]

        run_experiment_lite(
            run_task,
            use_cloudpickle=True,
            exp_prefix="fetch-sm-1-1",
            mode=MODE,
            n_parallel=0,
            env=env,
            seed=vv["seed"],
            snapshot_mode="last",
            variant=vv,
            terminate_machine=True,
            sync_all_data_node_to_s3=False,
            use_gpu=USE_GPU,
        )


if __name__ == "__main__":
    main()
