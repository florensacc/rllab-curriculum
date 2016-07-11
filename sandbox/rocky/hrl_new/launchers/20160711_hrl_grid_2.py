from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.hrl_new.algos.hrl_algos1 import HierPolopt
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.hrl_new.envs.image_grid_world import RandomImageGridWorld
from sandbox.rocky.hrl.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.hrl_new.policies.fixed_clock_policy import FixedClockPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

stub(globals())
from rllab.misc.instrument import VariantGenerator, variant


"""
Start adding MI bonus to new formulation
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [x * 100 + 11 for x in range(10)]

    @variant
    def setup(self):
        return ["hrl_new"]#"flat_policy", "hrl_trpo", "hrl_new"]

    @variant
    def mi_coeff(self):
        return [1.0]#0., 0.01, 0.1, 1.0, 10.0]

    @variant
    def subgoal_interval(self, setup):
        if setup == "flat_policy":
            return [None]
        return [3]#1, 3]

    @variant
    def env_size(self):
        return [5, 7, 9]

    @variant(hide=True)
    def env(self, env_size):
        action_map = [
            [0, 1, 1],  # Left
            [1, 3, 3],  # Down
            [2, 2, 0],  # Right
            [3, 0, 2],  # Up
        ]
        base_desc = [
            "." * env_size for _ in range(env_size)
        ]
        wrapped_env = RandomImageGridWorld(base_desc=base_desc)
        env = TfEnv(CompoundActionSequenceEnv(wrapped_env, action_map, obs_include_history=True))
        yield env

    @variant(hide=True)
    def policy(self, setup, env, subgoal_interval):
        if setup == "flat_policy":
            yield CategoricalMLPPolicy(
                env_spec=env.spec,
                name="policy"
            )
        else:
            yield FixedClockPolicy(
                env_spec=env.spec,
                subgoal_dim=50,
                bottleneck_dim=50,
                subgoal_interval=subgoal_interval,
                hidden_sizes=(32, 32),
                log_prob_tensor_std=1.0,
                name="policy"
            )

    @variant(hide=True)
    def algo(self, setup, env, policy, mi_coeff):
        baseline = LinearFeatureBaseline(env_spec=env.spec)

        if setup == "flat_policy":
            algo_cls = TRPO
            aux_policy = None
        elif setup == "hrl_trpo":
            algo_cls = TRPO
            aux_policy = None
        elif setup == "hrl_new":
            algo_cls = HierPolopt
            aux_policy = CategoricalMLPPolicy(
                env_spec=env.spec,
                name="aux_policy"
            )
        else:
            raise NotImplementedError

        algo = algo_cls(
            env=env,
            policy=policy,
            aux_policy=aux_policy,
            baseline=baseline,
            max_path_length=100,
            batch_size=10000,
            n_itr=500,
            discount=0.99,
            gae_lambda=0.99,
            mi_coeff=mi_coeff,
        )
        yield algo


variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants:
    run_experiment_lite(
        v["algo"].train(),
        exp_prefix="0711-hrl-grid-2",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        mode="local",
        variant=v,
    )
