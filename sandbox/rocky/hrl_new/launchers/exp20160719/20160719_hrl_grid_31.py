from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.hrl_new.envs.image_grid_world import ImageGridWorld
from sandbox.rocky.hrl.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from sandbox.rocky.hrl_new.launchers.exp20160718.algo2 import BonusTRPO, PredictionBonusEvaluator
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

stub(globals())
from rllab.misc.instrument import VariantGenerator, variant

"""
Try out naive prediction bonus on the hierarchical grid world task - a more challenging version
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [x * 100 + 11 for x in range(5)]

    @variant
    def setup(self):
        return ["flat_policy"]

    @variant
    def bonus_coeff(self):
        return [1.0, 0., 0.1, 0.01, 0.001]

    @variant
    def hidden_sizes(self):
        return [(32, 32)]

    @variant(hide=True)
    def env(self):
        action_map = [
            [0, 1, 1],  # Left
            [1, 3, 3],  # Down
            [2, 2, 0],  # Right
            [3, 0, 2],  # Up
        ]
        wrapped_env = ImageGridWorld(
            desc=[
                "Sx...",
                ".x.x.",
                ".x.x.",
                ".x.x.",
                "...xG",
            ]
        )
        env = TfEnv(CompoundActionSequenceEnv(wrapped_env, action_map, obs_include_history=True))
        yield env

    @variant(hide=True)
    def policy(self, setup, env, hidden_sizes):
        if setup == "flat_policy":
            yield CategoricalMLPPolicy(
                env_spec=env.spec,
                name="policy",
                hidden_sizes=hidden_sizes,
            )
        else:
            raise NotImplementedError

    @variant(hide=True)
    def algo(self, setup, env, policy, bonus_coeff):
        if setup == "flat_policy":
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            yield BonusTRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                batch_size=5000,
                n_itr=100,
                discount=0.99,
                gae_lambda=1.0,
                bonus_evaluator=PredictionBonusEvaluator(env.spec),
                bonus_coeff=bonus_coeff,
            )
        else:
            raise NotImplementedError


variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants:
    run_experiment_lite(
        v["algo"].train(),
        exp_prefix="0719-hrl-grid-31",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        mode="lab_kube",
        variant=v,
    )
