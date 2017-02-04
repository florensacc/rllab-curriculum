from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
import numpy as np
from rllab import config
import random

from sandbox.rocky.new_analogy.exp_utils import run_cirrascale, run_local_docker

"""
Use linear baseline, tweak discount
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0]#21]  # , 31]

    @variant
    def init_std(self):
        return [0.05, 0.1, 0.3, 0.5, 0.7, 1.0]

    @variant
    def discount(self):
        return [0.995, 0.99, 0.95, 0.9]


def run_task(vv):
    from sandbox.rocky.tf.algos.trpo import TRPO
    import tensorflow as tf
    from sandbox.rocky.s3.resource_manager import resource_manager
    from sandbox.rocky.new_analogy import fetch_utils

    import joblib
    with tf.Session() as sess:
        resource_name = "fetch_relative_dagger_pretrained_v1.pkl"

        file_name = resource_manager.get_file(resource_name=resource_name)

        policy = joblib.load(file_name)["policy"]

        horizon = 500

        env = fetch_utils.fetch_env()

        from sandbox.rocky.tf.baselines.linear_feature_baseline import LinearFeatureBaseline
        baseline = LinearFeatureBaseline(
            env_spec=env.spec,
        )

        sess.run(
            tf.assign(
                policy.wrapped_policy._l_std_param.param,
                [np.log(vv["init_std"])] * policy.wrapped_policy.action_space.flat_dim
            )
        )

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=horizon * 100,
            max_path_length=horizon,
            n_itr=5000,
            discount=vv["discount"],
            gae_lambda=0.97,
            parallel_vec_env=False,
            n_vectorized_envs=100,
        )

        algo.train(sess=sess)


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:
    run_cirrascale(
    # run_local_docker(
        run_task,
        exp_name="trpo-finetune-fetch-relative-1",
        variant=v,
        seed=v["seed"]
    )
