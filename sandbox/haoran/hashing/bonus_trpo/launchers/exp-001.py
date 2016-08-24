from __future__ import print_function
from __future__ import absolute_import

import os
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
        return [0.1, 0.01, 0.001, 0.]

    @variant
    def dim_key(self):
        return [64,256]

    @variant
    def discount(self):
        return [0.99]

    @variant
    def game(self):
        return ["breakout","freeway"] 

    @variant
    def bonus_form(self):
        return ["1/log(n+1)","1/n","1/sqrt(n)"]

    @variant
    def death_ends_episode(self):
        return [True,False]


variants = VG().variants()


print("#Experiments: %d" % len(variants))
exp_prefix = "bonus_trpo_atari/" + os.path.basename(__file__).split('.')[0]

for v in variants:
    env = TfEnv(AtariEnv(
            game=v["game"], 
            obs_type="ram",
            death_ends_episode=v["death_ends_episode"]
        ))
    policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32), name="policy")
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    bonus_baseline = LinearFeatureBaseline(env_spec=env.spec)
    bonus_evaluator = HashingBonusEvaluator(
        env_spec=env.spec, 
        dim_key=v["dim_key"],
        bonus_form=v["bonus_form"]
    )
    algo = BonusTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        bonus_evaluator=bonus_evaluator,
        bonus_baseline=bonus_baseline,
        bonus_coeff=v["bonus_coeff"],
        batch_size=5000,
        max_path_length=4500,
        discount=v["discount"],
        n_itr=1000,
        clip_reward=True,
    )

    import datetime
    import dateutil.tz
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    exp_name = "alex_{time}_{game}".format(
        time=timestamp,
        game=v["game"],
    )


    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        mode="ec2",
        variant=v,
    )

if "local" not in mode:
    os.system("chmod 444 %s"%(__file__))

