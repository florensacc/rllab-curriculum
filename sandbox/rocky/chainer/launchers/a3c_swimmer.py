from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite


def run_task(*_):
    from sandbox.rocky.chainer.algos.a3c import A3C
    from sandbox.rocky.chainer.actor_critics.gaussian_mlp_actor_critic import GaussianMLPActorCritic
    import chainer.functions as F

    env = normalize(SwimmerEnv())

    policy = GaussianMLPActorCritic(
        env_spec=env.spec,
        hidden_sizes=(256, 256, 256),
        hidden_nonlinearity=F.relu,
    )

    algo = A3C(
        env=env,
        policy=policy,
        n_envs_per_worker=1,
        update_frequency=100,
        learning_rate=1e-4,
        n_parallel=16,
        max_path_length=500,
        discount=0.99,
        gae_lambda=1.,
        epoch_length=10000,
        n_epochs=100,
        max_grad_norm=1000000.,
        scale_reward=1.,
        entropy_bonus_coeff=0,
        optimizer_type="adam",
        normalize_adv=True,
        normalize_vf=True,
        normalize_momentum=0.99,
        share_optimizer=True,
    )
    algo.train()


run_experiment_lite(
    run_task,
    mode="local_docker",
    use_cloudpickle=True,
    docker_image="dementrock/rllab3-shared-gpu-cuda80:20161120",
    seed=0,
    env=dict(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
)
