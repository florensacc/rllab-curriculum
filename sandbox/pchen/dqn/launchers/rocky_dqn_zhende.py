from __future__ import print_function
from __future__ import absolute_import

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.pchen.dqn.envs.atari import AtariEnvCX
from sandbox.pchen.dqn.q_functions.discrete_conv_q_function import DiscreteConvQFunction

from sandbox.rocky.tf.envs.base import TfEnv

from sandbox.rocky.arql.algos.dqn import DQN
from sandbox.rocky.arql.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

"""
Actual experiment with Autoregressive Q networks (or rather, autoregressive transformation on environments) with fixes
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 111,]

    # @variant
    # def n_bins(self):
    #     return [3, 5, 7, 9]

    @variant
    def rom(self):
        return ["breakout", "pong"]

    @variant(hide=True)
    def env(self, rom):
        yield TfEnv((normalize(AtariEnvCX(rom, obs_type='image'))))

    # @variant
    # def hidden_sizes(self):
    #     return [(100, 100)]
    #
    # @variant
    # def scale_reward(self):
    #     return [1., 0.1, 0.01]


variants = VG().variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    env = v["env"]
    qf = DiscreteConvQFunction(env_spec=env.spec, name="qf")
    target_qf = DiscreteConvQFunction(env_spec=env.spec, name="target_qf")
    es = EpsilonGreedyStrategy(env_spec=env.spec, epsilon_decay_range=1000000)
    algo = DQN(
        env=env,
        qf=qf,
        target_qf=target_qf,
        es=es,
        n_epochs=5000,
        epoch_length=1000,
        batch_size=32,
        max_path_length=500,
        discount=0.99 ** (1. ),
        eval_max_path_length=500 ,
        eval_samples=10000 ,
        min_pool_size=10000 ,
        replay_pool_size=1000000 ,
        target_update_interval=10000 ,
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="0901-atari",
        seed=v["seed"],
        n_parallel=4,
        # snapshot_mode="last",
        # mode="lab_kube",
        # variant=v,
        mode="local",
    )
