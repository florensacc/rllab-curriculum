from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
import numpy as np
from rllab import config
import random

from sandbox.rocky.new_analogy.exp_utils import run_cirrascale, run_local_docker, run_local

"""
Use linear baseline, tweak discount; use the fetch_rl environment
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0]

    @variant
    def init_mocap_std(self):
        return [0.01, 0.02, 0.03]#1e-5, 1e-4, 1e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2]

    @variant
    def init_gripper_std(self):
        return [0.01, 0.1, 0.2, 0.3, 0.5, 1.0]

    @variant
    def discount(self):
        return [0.995]#, 0.99, 0.95, 0.9]


def run_task(vv):
    from sandbox.rocky.tf.algos.trpo import TRPO
    import tensorflow as tf
    from sandbox.rocky.s3.resource_manager import resource_manager
    from sandbox.rocky.new_analogy import fetch_utils
    from gpr.worldgen.world import set_in_rl

    import joblib
    with tf.Session() as sess:
        resource_name = "fetch_relative_dagger_pretrained_v1.pkl"

        file_name = resource_manager.get_file(resource_name=resource_name)

        policy = joblib.load(file_name)["policy"]

        # set_in_rl()

        horizon = 500
        multiple = 100
        n_itr = 5000

        # horizon = 500
        # multiple = 10#0
        # n_itr = 1

        env = fetch_utils.fetch_env(usage="rl")

        # from sandbox.rocky.tf.baselines.linear_feature_baseline import LinearFeatureBaseline
        # baseline = LinearFeatureBaseline(
        #     env_spec=env.spec,
        # )

        from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
        from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                use_trust_region=True,
                hidden_sizes=(128, 128),
                optimizer=ConjugateGradientOptimizer(),
                step_size=0.1,
            ),
        )

        sess.run(
            tf.assign(
                policy.wrapped_policy._l_std_param.param,
                [np.log(vv["init_mocap_std"])] * 3 + [np.log(vv["init_gripper_std"])] + [-10] * 4,
            )
        )

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=horizon * multiple,
            max_path_length=horizon,
            n_itr=n_itr,
            discount=vv["discount"],
            gae_lambda=0.97,
            parallel_vec_env=False,
            n_vectorized_envs=multiple,
        )

        algo.train(sess=sess)


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:
    run_cirrascale(
        # run_local_docker(
        # run_local(
        run_task,
        exp_name="trpo-finetune-fetch-relative-2-1",
        variant=v,
        seed=v["seed"]
    )
