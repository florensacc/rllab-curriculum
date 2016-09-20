from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.neural_learner.algos.bonus_algos import BonusPPO
from sandbox.rocky.neural_learner.algos.pposgd import PPOSGD
from sandbox.rocky.neural_learner.bonus_evaluators.rnn_prediction_bonus_evaluator import RNNPredictionBonusEvaluator
from sandbox.rocky.tf.policies.categorical_rnn_policy import CategoricalRNNPolicy
from sandbox.rocky.neural_learner.envs.random_maze_env import RandomMazeEnv
from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv
from sandbox.rocky.neural_learner.envs.partial_obs_maze_env import PartialObsMazeEnv
from sandbox.rocky.neural_learner.envs.maze.empty_maze_generator import EmptyMazeGenerator
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.ppo import PPO
from sandbox.rocky.tf.algos.cem import CEM
import sandbox.rocky.tf.core.layers as L
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.optimizers.penalty_optimizer import PenaltyOptimizer
import tensorflow as tf
import sys

from sandbox.rocky.tf.policies.rnn_utils import NetworkType

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant


"""
First order algorithm for optimizing RNN policies
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]

    @variant
    def size(self):
        return [11]

    @variant
    def horizon(self, n_episodes):
        if n_episodes == 2:
            yield 100
        elif n_episodes == 10:
            yield 30
        else:
            raise NotImplementedError

    @variant
    def n_episodes(self):
        return [10]

    @variant
    def network_type(self):
        return [
            # NetworkType.PSEUDO_LSTM,
            # NetworkType.PSEUDO_LSTM_GATE_SQUASH,
            NetworkType.GRU,
            # NetworkType.TF_GRU,
            # NetworkType.TF_BASIC_LSTM,
            # NetworkType.LSTM,
            # NetworkType.LSTM_PEEPHOLE,
        ]

    @variant
    def batch_size(self):
        return [10000]  # , 30000]#, 50000, 100000]

    @variant
    def discount(self):
        return [0.99]

    @variant
    def hidden_dim(self):
        return [32]#, 100]

    @variant(hide=True)
    def env(self, size, n_episodes, discount, horizon):
        episode_env = PartialObsMazeEnv(
            RandomMazeEnv(
                n_row=size,
                n_col=size,
                maze_gen=EmptyMazeGenerator()
            )
        )
        yield TfEnv(MultiEnv(
            wrapped_env=episode_env, n_episodes=n_episodes, episode_horizon=horizon, discount=discount,
        ))

    # @variant
    # def algo_type(self):
    #     return ["pposgd"]  # "trpo", "pposgd", "ppo"]

    # @variant
    # def bonus_coeff(self):
    #     return [0.1, 0., 1., 10., 0.01, 0.001]


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    baseline = LinearFeatureBaseline(env_spec=v["env"].spec)

    policy = CategoricalRNNPolicy(
        name="policy",
        env_spec=v["env"].spec,
        hidden_dim=v["hidden_dim"],
        network_type=v["network_type"]
    )

    algo = PPOSGD(
        env=v["env"],
        policy=policy,
        baseline=baseline,
        max_path_length=v["horizon"] * v["n_episodes"],
        batch_size=v["batch_size"],
        discount=v["discount"],
        n_itr=1000,
        sampler_args=dict(n_envs=10),
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="rlrl-maze-20",
        mode="local",
        n_parallel=0,
        seed=v["seed"],
        variant=v,
        snapshot_mode="last",
    )
    sys.exit()
