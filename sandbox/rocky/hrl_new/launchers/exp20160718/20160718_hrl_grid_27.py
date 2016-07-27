from __future__ import absolute_import
from __future__ import print_function

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl_new.envs.image_grid_world import RandomImageGridWorld
from sandbox.rocky.hrl_new.launchers.exp20160718.algo import Algo, CategoricalLookbackPolicy
from sandbox.rocky.tf.envs.base import TfEnv

stub(globals())
from rllab.misc.instrument import VariantGenerator, variant

"""
Fancier optimization schemes
"""

N_ITR = 200



class VG(VariantGenerator):
    @variant
    def seed(self):
        return [x * 100 + 11 for x in range(5)]

    @variant
    def env_size(self):
        return [5]

    @variant
    def bottleneck_dim(self):
        return [32]#1, 2, 5, 10, 32]

    @variant
    def mi_coeff(self):
        return [0., 0.1, 0.5, 0.7, 0.9, 1.]#0., 0.01, 0.1, 1.]

    @variant(hide=True)
    def env(self, env_size):
        base_desc = [
            "." * env_size for _ in range(env_size)
            ]
        wrapped_env = RandomImageGridWorld(base_desc=base_desc)
        env = TfEnv(wrapped_env)
        yield env

    @variant(hide=True)
    def policy(self, env, bottleneck_dim=bottleneck_dim):
        yield CategoricalLookbackPolicy(
            env_spec=env.spec,
            bottleneck_dim=bottleneck_dim,
            name="policy"
        )

    @variant(hide=True)
    def algo(self, env, policy, mi_coeff):
        baseline = LinearFeatureBaseline(env_spec=env.spec)
        bonus_baseline = LinearFeatureBaseline(env_spec=env.spec)
        algo = Algo(
            env=env,
            policy=policy,
            baseline=baseline,
            bonus_baseline=bonus_baseline,
            max_path_length=10,
            batch_size=5000,
            n_itr=N_ITR,
            discount=0.99,
            gae_lambda=1.0,
            mi_coeff=mi_coeff,
            # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(symmetric=False, base_eps=1e-5)),
        )
        yield algo


variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants:
    run_experiment_lite(
        v["algo"].train(),
        exp_prefix="0718-hrl-grid-27",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        mode="lab_kube",
        variant=v,
    )
    # break
