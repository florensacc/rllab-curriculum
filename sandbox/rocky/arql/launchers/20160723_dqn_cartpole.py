


from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.rocky.arql.envs.discretized_env import DiscretizedEnv
from sandbox.rocky.arql.q_functions.discrete_mlp_q_function import DiscreteMLPQFunction
from sandbox.rocky.arql.algos.dqn import DQN
from sandbox.rocky.arql.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy
from sandbox.rocky.tf.envs.base import TfEnv

stub(globals())

"""
Experiment with new DQN implementation
"""

env = TfEnv(DiscretizedEnv(SwimmerEnv(), n_bins=3))
qf = DiscreteMLPQFunction(env_spec=env.spec, name="qf")
target_qf = DiscreteMLPQFunction(env_spec=env.spec, name="target_qf")
es = EpsilonGreedyStrategy(env_spec=env.spec, epsilon_decay_range=1000000)

algo = DQN(
    env=env,
    qf=qf,
    target_qf=target_qf,
    es=es,
    batch_size=32,
    discount=0.99,
    n_epochs=1000,
    epoch_length=1000,
    max_path_length=500,
    eval_samples=10000,
    scale_reward=0.1,
    eval_max_path_length=500,
    min_pool_size=10000,
    replay_pool_size=1000000,
)
run_experiment_lite(
    algo.train(),
    exp_prefix="0723-dqn-cartpole",
    seed=11,
    n_parallel=4,
    snapshot_mode="last",
    mode="local",
)
