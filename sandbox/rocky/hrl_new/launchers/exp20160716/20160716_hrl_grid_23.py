from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.hrl_new.envs.image_grid_world import RandomImageGridWorld
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from sandbox.rocky.hrl_new.policies.categorical_hgru_policy import CategoricalHGRUPolicy
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline


stub(globals())
from rllab.misc.instrument import VariantGenerator, variant

"""
Inspect GRU's behavior on simple grid
"""

N_ITR = 200


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [x * 100 + 11 for x in range(5)]

    @variant
    def setup(self):
        return ["gru_policy"]

    @variant
    def env_size(self):
        return [5]

    @variant(hide=True)
    def env(self, env_size):
        base_desc = [
            "." * env_size for _ in range(env_size)
            ]
        wrapped_env = RandomImageGridWorld(base_desc=base_desc)
        env = TfEnv(wrapped_env)
        yield env

    @variant(hide=True)
    def policy(self, env):
        yield CategoricalHGRUPolicy(
            env_spec=env.spec,
            name="policy"
        )

    @variant(hide=True)
    def algo(self, env, policy):
        baseline = LinearFeatureBaseline(env_spec=env.spec)
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=10,
            batch_size=5000,
            n_itr=N_ITR,
            discount=0.99,
            gae_lambda=1.0,
            optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(symmetric=False, base_eps=1e-5)),
        )
        yield algo


variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants:
    run_experiment_lite(
        v["algo"].train(),
        exp_prefix="0716-hrl-grid-23",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        mode="local",
        variant=v,
    )
    break
