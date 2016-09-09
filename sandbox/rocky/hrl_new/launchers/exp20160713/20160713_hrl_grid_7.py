


from sandbox.rocky.hrl_new.algos.hrl_algos2 import HierPolopt
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.hrl_new.envs.image_grid_world import RandomImageGridWorld
from sandbox.rocky.hrl.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.hrl_new.policies.fixed_clock_policy4 import FixedClockPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
# from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

stub(globals())
from rllab.misc.instrument import VariantGenerator, variant

"""
Baseline: train flat policy on non-compound version
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [x * 100 + 11 for x in range(5)]

    @variant
    def setup(self):
        return ["flat_policy"]

    @variant
    def env_size(self):
        return [5]

    @variant
    def hidden_sizes(self):
        return [(32, 32)]#, (100, 100), (300, 300), (300, 300, 300)]

    @variant(hide=True)
    def env(self, env_size):
        base_desc = [
            "." * env_size for _ in range(env_size)
            ]
        wrapped_env = RandomImageGridWorld(base_desc=base_desc)
        env = TfEnv(wrapped_env)
        yield env

    @variant(hide=True)
    def policy(self, env, hidden_sizes):
        yield CategoricalMLPPolicy(
            env_spec=env.spec,
            name="policy",
            hidden_sizes=hidden_sizes,
        )

    @variant(hide=True)
    def algo(self, env, policy):
        baseline = ZeroBaseline(env_spec=env.spec)
        algo_cls = TRPO
        algo = algo_cls(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=100/3,
            batch_size=5000/3,
            n_itr=100,
            discount=0.99,
            gae_lambda=1.0,
        )
        yield algo


variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants:
    run_experiment_lite(
        v["algo"].train(),
        exp_prefix="0713-hrl-grid-7",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        mode="local",
        variant=v,
    )
