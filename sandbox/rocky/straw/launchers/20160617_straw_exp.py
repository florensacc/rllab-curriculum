


# import os
#
# os.environ["THEANO_FLAGS"] = "device=cpu"

from sandbox.rocky.straw.policies.tf_straw_policy import STRAWPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.hrl.envs.straw_maze_env import STRAWMazeEnv
from rllab.envs.grid_world_env import GridWorldEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.core.network import MLP
# from rllab.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.straw.optimizers.tf_conjugate_gradient_optimizer import ConjugateGradientOptimizer, \
    FiniteDifferenceHvp
import tensorflow as tf

stub(globals())

env = TfEnv(STRAWMazeEnv())
env = TfEnv(GridWorldEnv())
# env = STRAWMazeEnv()
policy = STRAWPolicy(
    name="policy",
    env_spec=env.spec,
    feature_network_cls=MLP,
    sample_decision=True,
    feature_network_args=dict(
        name="feature_network",
        input_shape=(env.observation_space.flat_dim,),
        output_dim=32,
        hidden_sizes=(32,),
        hidden_nonlinearity=tf.nn.tanh,
        output_nonlinearity=tf.nn.tanh,
    )
)
# policy = CategoricalMLPPolicy(name="policy", env_spec=env.spec)

# algo = VPG(
#     env=env,
#     policy=policy,
#     baseline=LinearFeatureBaseline(env_spec=env.spec),
#     # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp()),
#     batch_size=1000,
#     max_path_length=50,
#     discount=0.99,
# )

algo = TRPO(
    env=env,
    policy=policy,
    baseline=LinearFeatureBaseline(env_spec=env.spec),
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp()),
    batch_size=1000,
    max_path_length=50,
    discount=0.99,
)

# policy.get_action(env.reset())


run_experiment_lite(
    algo.train(),
    n_parallel=1,
    seed=1,
    exp_prefix="straw",
    # env=dict(THEANO_FLAGS="optimizer=fast_compile"),
)
