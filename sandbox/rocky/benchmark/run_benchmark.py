from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.benchmark.trpo import TRPO

import gym

from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

spec = gym.benchmark_spec("Mujoco1M-v0")


def run_task(v):
    from sandbox.rocky.tf.envs.base import TfEnv
    from rllab.envs.normalized_env import normalize
    from rllab.envs.gym_env import GymEnv
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.benchmark.gaussian_mlp_baseline import GaussianMLPBaseline
    import tensorflow as tf

    env = TfEnv(normalize(GymEnv(v["env_id"], record_log=True, record_video=False, force_reset=False)))

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=tf.nn.tanh,
    )

    baseline = GaussianMLPBaseline(
        env_spec=env.spec,
        regressor_args=dict(
            use_trust_region=True,
            hidden_sizes=(64, 64),
            optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
            step_size=0.1,
            hidden_nonlinearity=tf.nn.tanh,
        ),
    )
    max_n_samples = v["max_timesteps"]
    batch_size = 1000
    n_itr = int(max_n_samples / batch_size)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=env.wrapped_env.wrapped_env.env.spec.timestep_limit,
        n_envs=10,
        n_itr=n_itr,
        parallel=False,
        step_size=0.003,
        discount=0.995,
        gae_lambda=0.97,
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
    )
    algo.train()


for task in spec.tasks:
    for seed in range(task.trials):
        run_experiment_lite(
            run_task,
            exp_prefix="mujoco-1m-trpo-tanh-1kbatch",
            snapshot_mode="last",
            sync_all_data_node_to_s3=False,
            mode=launch_cirrascale("pascal"),
            variant=dict(
                env_id=task.env_id,
                seed=seed,
                max_timesteps=task.max_timesteps,
                benchmark_id=spec.id,
            ),
            seed=seed,
            docker_image="dementrock/rllab3-shared-gpu-cuda80:20161130",
            use_gpu=True,
            use_cloudpickle=True,
        )
