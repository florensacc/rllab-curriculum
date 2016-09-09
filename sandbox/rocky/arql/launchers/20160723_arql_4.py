


from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.rocky.arql.envs.discretized_env import DiscretizedEnv
from sandbox.rocky.tf.envs.base import TfEnv

from sandbox.rocky.arql.q_functions.discrete_mlp_q_function import DiscreteMLPQFunction
from sandbox.rocky.arql.algos.dqn import DQN
from sandbox.rocky.arql.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

"""
Actual experiment with Autoregressive Q networks (or rather, autoregressive transformation on environments)
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 111, 211, 311, 411]

    @variant
    def n_bins(self):
        return [3, 5, 7, 9]

    @variant
    def env_cls(self):
        return [AntEnv, SwimmerEnv, HopperEnv, HalfCheetahEnv, Walker2DEnv]

    @variant(hide=True)
    def env(self, n_bins, env_cls):
        yield TfEnv(DiscretizedEnv(normalize(env_cls()), n_bins=n_bins))

    @variant
    def hidden_sizes(self):
        return [(100, 100)]

    @variant
    def scale_reward(self):
        return [1., 0.1, 0.01]


variants = VG().variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    env = v["env"]
    qf = DiscreteMLPQFunction(env_spec=env.spec, name="qf")
    target_qf = DiscreteMLPQFunction(env_spec=env.spec, name="target_qf")
    mult = env.wrapped_env.wrapped_env.action_dim
    es = EpsilonGreedyStrategy(env_spec=env.spec, epsilon_decay_range=1000000 * mult)
    algo = DQN(
        env=env,
        qf=qf,
        target_qf=target_qf,
        es=es,
        n_epochs=5000,
        epoch_length=1000 * mult,
        batch_size=32,
        scale_reward=v["scale_reward"],
        max_path_length=500 * mult,
        discount=0.99 ** (1. / mult),
        eval_max_path_length=500 * mult,
        eval_samples=10000 * mult,
        min_pool_size=10000 * mult,
        replay_pool_size=1000000 * mult,
        target_update_interval=10000 * mult,
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="0723-arql-4",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        mode="lab_kube",
        variant=v,
    )
