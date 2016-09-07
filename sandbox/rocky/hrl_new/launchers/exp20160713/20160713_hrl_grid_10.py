


from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.hrl_new.envs.image_grid_world import RandomImageGridWorld
from sandbox.rocky.hrl_new.algos.hrl_algos3 import TRPO
from sandbox.rocky.hrl.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from sandbox.rocky.hrl_new.policies.fixed_clock_policy5 import FixedClockPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
# from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

stub(globals())
from rllab.misc.instrument import VariantGenerator, variant

"""
#9 with more iterations
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [x * 100 + 11 for x in range(5)]

    @variant
    def setup(self):
        return ["hrl_trpo"]#, "flat_policy"]

    @variant
    def subgoal_interval(self, setup):
        if setup == "flat_policy":
            return [None]
        return [1, 3]

    @variant
    def env_size(self):
        return [5]

    @variant
    def hidden_sizes(self):
        return [(32, 32), (300, 300)]

    @variant
    def subgoal_dim(self, setup):
        if setup == "flat_policy":
            return [None]
        return [4, 10, 50]

    @variant
    def bottleneck_dim(self, setup):
        if setup == "flat_policy":
            return [None]
        return [3, 10, 50]

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
    def policy(self, setup, env, subgoal_interval, hidden_sizes, bottleneck_dim, subgoal_dim):
        if setup == "flat_policy":
            yield CategoricalMLPPolicy(
                env_spec=env.spec,
                name="policy",
                hidden_sizes=hidden_sizes,
            )
        else:
            yield FixedClockPolicy(
                env_spec=env.spec,
                subgoal_dim=subgoal_dim,
                bottleneck_dim=bottleneck_dim,
                subgoal_interval=subgoal_interval,
                hidden_sizes=hidden_sizes,
                log_prob_tensor_std=1.0,
                name="policy"
            )

    @variant(hide=True)
    def algo(self, env, policy):
        baseline = ZeroBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            batch_size=5000,
            n_itr=500,
            discount=0.99,
            gae_lambda=1.0,
        )
        yield algo


variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants:
    run_experiment_lite(
        v["algo"].train(),
        exp_prefix="0713-hrl-grid-10",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        mode="lab_kube",
        variant=v,
    )
    # break
