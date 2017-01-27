
from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.new_analogy.policies.normalizing_policy import NormalizingPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc import logger
import numpy as np
from rllab import config
import random

"""
Use linear baseline, tweak discount
"""

class VG(VariantGenerator):
    @variant
    def seed(self):
        return [21, 31]

    @variant
    def init_std(self):
        return [0.05]#, 0.1, 0.3, 0.5, 0.7, 1.0]

    @variant
    def discount(self):
        return [0.995, 0.99, 0.95, 0.9]


def run_task(vv):
    from sandbox.rocky.tf.algos.trpo import TRPO
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    import tensorflow as tf
    from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
    from sandbox.rocky.s3.resource_manager import resource_manager

    from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
    import joblib

    with tf.Session() as sess:
        resource_name = "stack_bc_v0.pkl"

        file_name = resource_manager.get_file(resource_name=resource_name)

        policy = joblib.load(file_name)["policy"]

        env = TfEnv(
            GprEnv(
                env_name="stack",
                experiment_args=dict(
                    nboxes=2,
                    horizon=300
                ),
                task_id=[[1, "top", 0]]
            )
        )

        from sandbox.rocky.tf.baselines.linear_feature_baseline import LinearFeatureBaseline
        baseline = LinearFeatureBaseline(
            env_spec=env.spec,
        )

        sess.run(tf.assign(policy._l_std_param.param, [np.log(vv["init_std"])] * env.action_dim))

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=60000,
            max_path_length=300,
            n_itr=5000,
            discount=vv["discount"],
            gae_lambda=0.97,
            parallel_vec_env=False,
            n_vectorized_envs=200,
        )

        algo.train(sess=sess)


variants = VG().variants()

print("#Experiments:", len(variants))

env_args = dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package")
# mode = "local_docker"
mode = "ec2"
# mode = launch_cirrascale("pascal")
use_gpu = False
if mode == "local_docker":
    env_args["CUDA_VISIBLE_DEVICES"] = "1"

for v in variants:
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
        exp_prefix="trpo-finetune-stack-3-1",
        mode=mode,
        use_gpu=use_gpu,
        snapshot_mode="last",
        sync_all_data_node_to_s3=False,
        n_parallel=1,
        env=env_args,
        variant=v,
        seed=v["seed"],
    )
