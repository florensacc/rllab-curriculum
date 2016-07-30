from __future__ import print_function
from __future__ import absolute_import

from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.rocky.arql.envs.discretized_env import DiscretizedEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

"""
Investigate the effect of policy network size
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 111, 211, 311, 411]

    @variant
    def discretize(self):
        return [True]#, False]

    @variant
    def n_bins(self, discretize):
        if discretize:
            return [9]#3, 5, 7, 9]
        else:
            return [None]

    @variant
    def env_cls(self):
        return [Walker2DEnv, AntEnv]#, SimpleHumanoidEnv]

    @variant(hide=True)
    def env(self, discretize, n_bins, env_cls):
        if discretize:
            yield TfEnv(DiscretizedEnv(normalize(env_cls()), n_bins=n_bins))
        else:
            yield TfEnv(normalize(env_cls()))

    @variant
    def hidden_sizes(self):
        return [(32, 32), (128, 128), (256, 256)]


variants = VG().variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    env = v["env"]
    if v["discretize"]:
        policy = CategoricalMLPPolicy(env_spec=env.spec, name="policy", hidden_sizes=v["hidden_sizes"])
    else:
        policy = GaussianMLPPolicy(env_spec=env.spec, name="policy", hidden_sizes=v["hidden_sizes"])
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    if v["discretize"]:
        mult = env.wrapped_env.wrapped_env.action_dim
    else:
        mult = 1
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=50000 * mult,
        max_path_length=500 * mult,
        discount=0.99 ** (1. / mult),
        step_size=0.01,
        n_itr=500,
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="0723-arql-3",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        mode="lab_kube",
        variant=v,
    )
