from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite


def run_task(*_):
    from sandbox.rocky.chainer.algos.a3c import A3C
    from sandbox.rocky.chainer.actor_critics.gaussian_mlp_actor_critic import GaussianMLPActorCritic
    import chainer.functions as F

    env = normalize(CartpoleEnv())

    policy = GaussianMLPActorCritic(
        env_spec=env.spec,
        hidden_sizes=(100, 100),
        hidden_nonlinearity=F.relu,
    )

    algo = A3C(
        env=env,
        policy=policy,
        n_envs_per_worker=1,
        update_frequency=20,
        learning_rate=1e-3,
        n_parallel=16,
        max_path_length=500,
        discount=0.995,
        gae_lambda=0.97,
        epoch_length=10000,
        n_epochs=1000,
        scale_reward=0.01,#1.,#0.01,
        normalize_adv=True,#False,
        normalize_vf=True,#False,
    )
    algo.train()


run_experiment_lite(
    run_task,
    mode="local_docker",
    use_cloudpickle=True,
    docker_image="dementrock/rllab3-shared-gpu-cuda80:20161120",
    env=dict(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
)
