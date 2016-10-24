from rllab.misc.instrument import stub, run_experiment_lite
# from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
# from sandbox.rocky.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
# from sandbox.rocky.tf.policies.categorical_lstm_policy import CategoricalLSTMPolicy
from sandbox.rocky.tf.policies.categorical_rnn_policy import CategoricalRNNPolicy
from sandbox.rocky.neural_learner.envs.random_maze_env import RandomMazeEnv
from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv
from sandbox.rocky.neural_learner.envs.partial_obs_maze_env import PartialObsMazeEnv
from sandbox.rocky.neural_learner.envs.maze.empty_maze_generator import EmptyMazeGenerator
from sandbox.rocky.neural_learner.envs.maze.dfs_maze_generator import DFSGridMazeGenerator
from sandbox.rocky.neural_learner.envs.choice_env import ChoiceEnv
from rllab.envs.grid_world_env import GridWorldEnv
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
Fix PseudoLSTM bug
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]  # , 41, 51]

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
        return [2, 10]

    @variant
    def network_type(self):
        return [
            NetworkType.PSEUDO_LSTM,
            NetworkType.PSEUDO_LSTM_GATE_SQUASH,
            # "pseudo_lstm",
            # "pseudo_lstm_gatesquash",
            # "lstm",
            # "lstm_peephole",
            # "gru",
            # "tf_basic_lstm",
            # "tf_gru",
        ]

    @variant
    def batch_size(self):
        return [10000]  # , 30000]#, 50000, 100000]

    @variant
    def discount(self):
        return [0.99]

    @variant
    def hidden_dim(self):
        return [32, 100]

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

    @variant
    def algo_type(self):
        return ["ppo"]  # "trpo", "pposgd", "ppo"]


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

    if v["algo_type"] == "trpo":
        algo = TRPO(
            env=v["env"],
            policy=policy,
            baseline=baseline,
            max_path_length=v["horizon"] * v["n_episodes"],
            batch_size=v["batch_size"],
            discount=v["discount"],
            n_itr=1000,
            sampler_args=dict(n_envs=10),
            optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
        )
    elif v["algo_type"] == "pposgd":
        algo = PPO(
            env=v["env"],
            policy=policy,
            baseline=baseline,
            max_path_length=v["horizon"] * v["n_episodes"],
            batch_size=v["batch_size"],
            discount=v["discount"],
            n_itr=1000,
            sampler_args=dict(n_envs=10),
            optimizer=PenaltyOptimizer(
                FirstOrderOptimizer(
                    tf_optimizer_cls=tf.train.AdamOptimizer,
                    tf_optimizer_args=dict(learning_rate=1e-3),
                    max_epochs=10,
                    batch_size=10,
                ),
                initial_penalty=1.,
                data_split=0.9,  # None,#0.5,
                max_penalty=1e4,
                adapt_penalty=True,
                max_penalty_itr=3,
                increase_penalty_factor=1.5,
                decrease_penalty_factor=1. / 1.5,
                barrier_coeff=1e2,
            )
            # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
        )
    elif v["algo_type"] == "ppo":
        algo = PPO(
            env=v["env"],
            policy=policy,
            baseline=baseline,
            max_path_length=v["horizon"] * v["n_episodes"],
            batch_size=v["batch_size"],
            discount=v["discount"],
            n_itr=1000,
            sampler_args=dict(n_envs=10),
            # optimizer=PenaltyOptimizer(
            #     FirstOrderOptimizer(
            #         tf_optimizer_cls=tf.train.AdamOptimizer,
            #         tf_optimizer_args=dict(learning_rate=1e-3),
            #         max_epochs=10,
            #         batch_size=10,
            #     ),
            #     initial_penalty=1.,
            #     data_split=0.9,#None,#0.5,
            #     max_penalty=1e4,
            #     adapt_penalty=True,
            #     max_penalty_itr=3,
            #     increase_penalty_factor=1.5,
            #     decrease_penalty_factor=1./1.5,
            #     barrier_coeff=1e2,
            # )
            # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
        )
    else:
        raise NotImplementedError

    run_experiment_lite(
        algo.train(),
        exp_prefix="rlrl-maze-13",
        mode="lab_kube",
        n_parallel=0,
        seed=v["seed"],
        variant=v,
        snapshot_mode="last",
    )
    # sys.exit()
