from __future__ import absolute_import
from __future__ import print_function

# import os
#
# os.environ["THEANO_FLAGS"] = "device=cpu"

from sandbox.rocky.straw.policies.tf_straw_policy import STRAWPolicy
from sandbox.rocky.hrl.envs.straw_maze_env import STRAWMazeEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.vpg import VPG
# from rllab.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.straw.optimizers.tf_conjugate_gradient_optimizer import ConjugateGradientOptimizer, \
    FiniteDifferenceHvp

stub(globals())

env = TfEnv(STRAWMazeEnv())
# env = STRAWMazeEnv()
policy = STRAWPolicy(name="policy", env_spec=env.spec)

# algo = VPG(
#     env=env,
#     policy=policy,
#     baseline=LinearFeatureBaseline(env_spec=env.spec),
#     # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp()),
#     batch_size=100,
#     max_path_length=100,
#     discount=0.99,
# )

algo = TRPO(
    env=env,
    policy=policy,
    baseline=LinearFeatureBaseline(env_spec=env.spec),
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp()),
    batch_size=100,
    max_path_length=100,
    discount=0.99,
)

# policy.get_action(env.reset())


run_experiment_lite(
    algo.train(),
    n_parallel=1,
    exp_prefix="straw",
    # env=dict(THEANO_FLAGS="optimizer=fast_compile"),
)

