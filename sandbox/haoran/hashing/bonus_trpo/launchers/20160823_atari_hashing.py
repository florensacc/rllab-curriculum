from __future__ import print_function
from __future__ import absolute_import

from sandbox.haoran.hashing.algos.bonus_trpo import BonusTRPO
from sandbox.haoran.hashing.bonus_evaluators.hashing_bonus_evaluator import HashingBonusEvaluator
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.haoran.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.haoran.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.haoran.hashing.envs.atari import AtariEnv
from rllab import config

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

"""
Fix to counting scheme. Fix config...
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [311, 411, 511, 611, 711]

    @variant
    def bonus_coeff(self):
        return [0., 0.1, 0.01, 0.001]

    @variant
    def dim_key(self):
        return [64, 256]

    @variant
    def discount(self):
        return [0.99]

    @variant
    def game(self):
        return ["solaris","hero","wizard_of_wor","amidar","alien"]


variants = VG().variants()


print("#Experiments: %d" % len(variants))

for v in variants:
    env = TfEnv(AtariEnv(game=v["game"], obs_type="ram"))
    policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32), name="policy")
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    bonus_baseline = LinearFeatureBaseline(env_spec=env.spec)
    bonus_evaluator = HashingBonusEvaluator(env_spec=env.spec, dim_key=v["dim_key"])
    algo = BonusTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        bonus_evaluator=bonus_evaluator,
        bonus_baseline=bonus_baseline,
        bonus_coeff=v["bonus_coeff"],
        batch_size=50000,
        max_path_length=4500,
        discount=v["discount"],
        n_itr=1000,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="0823-atari-hashing",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        mode="ec2",
        variant=v,
    )
