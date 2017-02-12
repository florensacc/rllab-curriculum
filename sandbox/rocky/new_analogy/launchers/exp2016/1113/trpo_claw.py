# First experiment with claw, using behavior cloning
# The end goal is to get some reasonable behavior using as few demonstrations as possible
from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc import logger
import numpy as np


def run_task(*_):
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.tf.algos.trpo import TRPO
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    import tensorflow as tf

    from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline

    from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

    # logger.log("Loading data...")
    # data = np.load("/shared-data/claw-2k-data.npz")
    # exp_x = data["exp_x"]
    # exp_u = data["exp_u"]
    # exp_rewards = data["exp_rewards"]
    # logger.log("Loaded")
    # paths = []

    # for xs, us, rewards in zip(exp_x, exp_u, exp_rewards):
    #     paths.append(dict(observations=xs, actions=us, rewards=rewards))

    env = TfEnv(GprEnv("TF2"))#, fixed_reset=True))

    baseline = GaussianMLPBaseline(
        env_spec=env.spec,
        regressor_args=dict(
            use_trust_region=True,
            hidden_sizes=(128, 128),
            hidden_nonlinearity=tf.nn.relu,
            optimizer=ConjugateGradientOptimizer(),
            step_size=0.1,
        ),
    )

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(128, 128),
        hidden_nonlinearity=tf.nn.relu,
        name="policy"
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
        parallel_vec_env=True,
        n_vectorized_envs=100,
    )

    algo.train()


for seed in [11, 21, 31, 41, 51]:

    if __name__ == "__main__":
        run_experiment_lite(
            run_task,
            use_cloudpickle=True,
            exp_prefix="trpo-claw-1",
            mode=launch_cirrascale("pascal"),
            use_gpu=True,
            snapshot_mode="last",
            sync_all_data_node_to_s3=False,
            n_parallel=8,
            env=dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/conopt_root"),
            docker_image="quay.io/openai/rocky-rllab3-conopt-gpu-pascal",
            docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
            variant=dict(seed=seed),
            seed=seed,
        )
