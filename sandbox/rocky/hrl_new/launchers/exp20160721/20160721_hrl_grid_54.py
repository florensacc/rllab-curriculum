


from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.hrl_new.envs.image_grid_world import ImageGridWorld
from sandbox.rocky.hrl.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from sandbox.rocky.hrl_new.launchers.exp20160721.algo1 import BonusTRPO, BottleneckStateGoalSurpriseBonusEvaluator, \
    FixedClockPolicy
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

stub(globals())
from rllab.misc.instrument import VariantGenerator, variant

"""
Use k-step prediction for subgoals, where k is the subgoal interval
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [x * 100 + 11 for x in range(10)]

    @variant
    def setup(self):
        return ["hrl_policy"]

    @variant
    def bonus_coeff(self):
        return [0.0001]

    @variant
    def subgoal_dim(self, setup):
        if setup == "hrl_policy":
            return [10]  # , 20]#5, 10, 64]
        raise NotImplementedError

    @variant
    def regressor_use_trust_region(self, setup):
        if setup == "hrl_policy":
            return [False]  # True]#, False]
        raise NotImplementedError

    @variant
    def mi_coeff(self):
        return [1., 0.]

    @variant
    def surprise_coeff(self):
        return [0., 1.]

    @variant
    def subgoal_interval(self):
        return [3, 2, 1, 4]

    @variant
    def predict_interval_ahead(self):
        return [True, False]

    @variant
    def hidden_sizes(self):
        return [(32, 32)]

    @variant
    def env_mode(self):
        return ["easy", "medium", "hard"]

    @variant(hide=True)
    def env(self, env_mode):
        action_map = [
            [0, 1, 1],  # , 2, 1, 3],  # Left
            [1, 3, 3],  # , 0, 2, 2],  # Down
            [2, 2, 0],  # , 1, 3, 0],  # Right
            [3, 0, 2],  # , 3, 0, 1],  # Up
        ]
        if env_mode == "easy":
            desc = [
                "S....",
                ".....",
                ".....",
                ".....",
                "....G",
            ]
        elif env_mode == "medium":
            desc = [
                "..x..",
                "..x..",
                ".....",
                "..x..",
                "S.x.G",
            ]
        elif env_mode == "hard":
            desc = [
                "Sx...",
                ".x.x.",
                ".x.x.",
                ".x.x.",
                "...xG",
            ]
        else:
            raise NotImplementedError
        wrapped_env = ImageGridWorld(desc=desc)
        env = TfEnv(CompoundActionSequenceEnv(wrapped_env, action_map, obs_include_history=True))
        yield env

    @variant(hide=True)
    def policy(self, setup, env, hidden_sizes, subgoal_dim, subgoal_interval):
        if setup == "flat_policy":
            yield CategoricalMLPPolicy(
                env_spec=env.spec,
                name="policy",
                hidden_sizes=hidden_sizes,
            )
        elif setup == "hrl_policy":
            yield FixedClockPolicy(
                env_spec=env.spec,
                name="policy",
                hidden_sizes=hidden_sizes,
                subgoal_interval=subgoal_interval,
                subgoal_dim=subgoal_dim,
            )
        else:
            raise NotImplementedError

    @variant(hide=True)
    def algo(self, setup, env, policy, bonus_coeff, regressor_use_trust_region, subgoal_dim, mi_coeff,
             surprise_coeff, subgoal_interval, predict_interval_ahead):
        if setup == "flat_policy" or setup == "hrl_policy":
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            yield BonusTRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                batch_size=5000,
                n_itr=500,
                discount=0.99,
                gae_lambda=1.0,
                bonus_evaluator=BottleneckStateGoalSurpriseBonusEvaluator(
                    env.spec,
                    subgoal_dim=subgoal_dim,
                    subgoal_interval=subgoal_interval,
                    mi_coeff=mi_coeff,
                    surprise_coeff=surprise_coeff,
                    predict_interval_ahead=predict_interval_ahead,
                    regressor_args=dict(
                        use_trust_region=regressor_use_trust_region,
                        hidden_sizes=(200, 200),
                    ),
                    max_epochs=5,
                ),
                bonus_coeff=bonus_coeff,
            )
        else:
            raise NotImplementedError


variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants:
    run_experiment_lite(
        v["algo"].train(),
        exp_prefix="0721-hrl-grid-54",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        mode="lab_kube",
        variant=v,
    )
    # break
