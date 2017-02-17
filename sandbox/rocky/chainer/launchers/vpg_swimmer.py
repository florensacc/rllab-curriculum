from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite


def run_task(*_):
    from sandbox.rocky.chainer.algos.vpg import VPG
    from sandbox.rocky.chainer.policies.gaussian_mlp_policy import GaussianMLPPolicy

    # if __name__ == "__main__":
    # env = normalize(CartpoleEnv())
    #
    env = normalize(SwimmerEnv())

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(100, 100),#32, 32)
    )

    baseline = ZeroBaseline(env.spec)#LinearFeatureBaseline(env_spec=env.spec)

    algo = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=10000,
        max_path_length=500,
        n_itr=40,
        discount=0.99,
        learning_rate=0.001,
    )
    algo.train()


run_experiment_lite(
    run_task,
    mode="local_docker",
    use_cloudpickle=True,
    docker_image="dementrock/rllab3-shared-gpu-cuda80:20161120",
    seed=0,
    # env=dict(
    #     OMP_NUM_THREADS="1",
    #     MKL_NUM_THREADS="1",
    #     NUMEXPR_NUM_THREADS="1",
    # )
)
