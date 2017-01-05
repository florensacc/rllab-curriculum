import tensorflow as tf
from sandbox.haoran.mddpg.algos.ddpg import DDPG
from sandbox.haoran.mddpg.policies.nn_policy import FeedForwardPolicy
from sandbox.haoran.mddpg.qfunctions.nn_qfunction import FeedForwardCritic
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.rocky.tf.envs.base import TfEnv


def main():
    stub(globals())
    env = TfEnv(HalfCheetahEnv())
    es = OUStrategy(env_spec=env.spec)
    qf = FeedForwardCritic(
        "critic",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
    )
    policy = FeedForwardPolicy(
        "actor",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
    )
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
    )

    run_experiment_lite(
        algorithm.train(),
        n_parallel=1,
        snapshot_mode="last",
        exp_prefix="ddpg-half-cheetah",
        seed=2,
    )

if __name__ == "__main__":
    main()
