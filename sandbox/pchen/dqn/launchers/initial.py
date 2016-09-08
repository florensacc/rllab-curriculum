


from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.pchen.dqn.dqn_ported import DQNP
from sandbox.pchen.dqn.envs.atari import AtariEnvCX
from sandbox.pchen.dqn.q_functions.discrete_conv_q_function import DiscreteConvQFunction

from sandbox.rocky.tf.envs.base import TfEnv

from sandbox.rocky.arql.algos.dqn import DQN
from sandbox.rocky.arql.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy

from rllab import config
config.DOCKER_IMAGE = "dementrock/rllab-gpu:latest"

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

"""
Actual experiment with Autoregressive Q networks (or rather, autoregressive transformation on environments) with fixes
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 42]

    # @variant
    # def n_bins(self):
    #     return [3, 5, 7, 9]

    @variant
    def rom(self):
        return ["breakout", "pong"]

    @variant(hide=True)
    def env(self, rom):
        yield (((AtariEnvCX(rom, obs_type='image', life_terminating=True))))

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
    algo = DQNP(env)
    run_experiment_lite(
        algo.train(),
        exp_prefix="0901-atari-initial-gpu",
        seed=v["seed"],
        # n_parallel=4,
        snapshot_mode="last",
        mode="lab_kube",
        variant=v,
        use_gpu=True,
        # mode="local",
    )
