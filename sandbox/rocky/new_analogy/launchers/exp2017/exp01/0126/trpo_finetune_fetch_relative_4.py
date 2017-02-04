from rllab.misc.instrument import variant, VariantGenerator

from sandbox.rocky.new_analogy.exp_utils import run_cirrascale
from sandbox.rocky.new_analogy.tf.policies.auto_mlp_policy import AutoMLPPolicy

"""
Run TRPO without polishing its softmax temperature first
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0]#, 100, 200]

    @variant
    def discount(self):
        return [0.995]


def run_task(vv):
    from sandbox.rocky.tf.algos.trpo import TRPO
    import tensorflow as tf
    from sandbox.rocky.s3.resource_manager import resource_manager
    from sandbox.rocky.new_analogy import fetch_utils

    import joblib
    with tf.Session() as sess:
        resource_name = "fetch_5_blocks_dagger.pkl"
        file_name = resource_manager.get_file(resource_name=resource_name)
        policy = joblib.load(file_name)["policy"]
        assert isinstance(policy, fetch_utils.DiscretizedFetchWrapperPolicy)

        # resource_name = "fetch_5_blocks_100_paths.pkl"
        # file_name = resource_manager.get_file(resource_name=resource_name)
        # paths_data = joblib.load(file_name)

        multiple = 100
        n_itr = 5000
        horizon = 2000
        n_boxes = 5

        env = fetch_utils.fetch_env(usage="rl", horizon=horizon, height=n_boxes)
        env = fetch_utils.DiscretizedEnvWrapper(wrapped_env=env, disc_intervals=policy.disc_intervals)

        nn_policy = policy.wrapped_policy
        assert isinstance(nn_policy, AutoMLPPolicy)

        # env.step(nn_policy.get_action(env.reset())[0])
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

        algo = TRPO(
            env=env,
            policy=nn_policy,
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
    # run_ec2(
    # run_local(
        # run_local_docker(
        run_task,
        exp_name="trpo-finetune-fetch-relative-4",
        variant=v,
        seed=v["seed"],
        n_parallel=0,
    )
