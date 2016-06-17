from __future__ import absolute_import
from __future__ import print_function

# import os

# os.environ["THEANO_FLAGS"] = "optimizer=fast_compile"

from sandbox.rocky.straw.policies.straw_policy import STRAWPolicy
from sandbox.rocky.hrl.envs.straw_maze_env import STRAWMazeEnv
# from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.algos.trpo import TRPO
# from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

stub(globals())

# env = TfEnv(STRAWMazeEnv())
env = STRAWMazeEnv()
policy = STRAWPolicy(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=LinearFeatureBaseline(env_spec=env.spec),
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp()),
)

# policy.get_action(env.reset())


run_experiment_lite(
    algo.train(),
#     env=dict(THEANO_FLAGS="mode=FAST_COMPILE,optimizer=None"),
)

