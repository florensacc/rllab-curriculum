from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.hrl_new.algos.hrl_algos1 import HierPolopt
from sandbox.rocky.hrl.envs.atari import AtariEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.hrl_new.policies.fixed_clock_policy import FixedClockPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

stub(globals())
from rllab.misc.instrument import VariantGenerator, variant

"""
Rerun with more logging
Comparison between:
- Using flat policy throughout
- Using hierarchical policy, train using usual TRPO
- Using hierarchical policy, train using new strategy
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [x * 100 + 11 for x in range(10)]

    @variant
    def game(self):
        return ["freeway", "breakout", "frostbite"]

    @variant
    def setup(self):
        return ["flat_policy", "hrl_trpo", "hrl_new"]

    @variant
    def subgoal_interval(self, setup):
        if setup == "flat_policy":
            return [None]
        return [1, 3, 10]

    @variant(hide=True)
    def env(self, game):
        yield TfEnv(AtariEnv(game=game, obs_type="ram", frame_skip=4))

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
    def algo(self, setup, env, policy):
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
            max_path_length=4500,
            batch_size=50000,
            n_itr=500,
            discount=0.99,
            gae_lambda=0.99,
            mi_coeff=0.,
        )
        yield algo

variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants:
    run_experiment_lite(
        v["algo"].train(),
        exp_prefix="0710-hrl-atari-ram-13",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        mode="lab_kube",
        variant=v,
    )
