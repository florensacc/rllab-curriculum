import tensorflow as tf
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.rocky.tf.algos.ddpg import DDPG as DDPG
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.deterministic_mlp_policy import \
    DeterministicMLPPolicy
from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import \
    ContinuousMLPQFunction


def main():
    stub(globals())
    ddpg_params = dict(
        batch_size=64,
        n_epochs=2000,
        epoch_length=1000,
        eval_samples=1000,
        discount=0.99,
        qf_learning_rate=1e-3,
        policy_learning_rate=1e-4,
        soft_target_tau=0.001,
        replay_pool_size=1000000,
        min_pool_size=1000,
        scale_reward=0.1,
    )
    env = TfEnv(HalfCheetahEnv())
    es = OUStrategy(env_spec=env.spec)

    policy = DeterministicMLPPolicy(
        name="init_policy",
        env_spec=env.spec,
        hidden_sizes=(100, 100),
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
    )
    qf = ContinuousMLPQFunction(
        name="qf",
        env_spec=env.spec,
        hidden_sizes=(100, 100),
        bn=False,
    )

    algorithm = DDPG(
        env,
        policy,
        qf,
        es,
        **ddpg_params
    )

    run_experiment_lite(
        algorithm.train(),
        n_parallel=1,
        snapshot_mode="last",
        exp_prefix="ddpg-shane-half-cheetah-script",
        seed=1,
        variant=ddpg_params,
    )

if __name__ == "__main__":
    main()
