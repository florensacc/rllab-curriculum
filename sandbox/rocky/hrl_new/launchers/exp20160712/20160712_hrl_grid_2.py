


from sandbox.rocky.hrl_new.algos.hrl_algos2 import HierPolopt
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.hrl_new.envs.image_grid_world import RandomImageGridWorld
from sandbox.rocky.hrl.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.hrl_new.policies.fixed_clock_policy2 import FixedClockPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
# from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

stub(globals())
from rllab.misc.instrument import VariantGenerator, variant

"""
Proper implementation of imitation part
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [x * 100 + 11 for x in range(5)]

    @variant
    def setup(self):
        return ["flat_policy", "hrl_new"]

    @variant
    def mi_coeff(self, setup):
        if setup == "flat_policy":
            return [None]
        return [0., 0.1, 1.0, 10.]

    @variant
    def subgoal_interval(self, setup):
        if setup == "flat_policy":
            return [None]
        return [1, 3]

    @variant
    def env_size(self):
        return [5]

    @variant
    def bonus_upscale_coeff(self, setup):
        if setup == "flat_policy":
            return [None]
        return [1., 2., 5.]

    @variant
    def imitation_max_epochs(self, setup):
        if setup == "flat_policy":
            return [None]
        return [3, 5, 10]

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
                name="policy",
                hidden_sizes=(300, 300),
            )
        else:
            yield FixedClockPolicy(
                env_spec=env.spec,
                subgoal_dim=4,
                bottleneck_dim=3,
                subgoal_interval=subgoal_interval,
                hidden_sizes=(300, 300),
                # hidden_sizes=(32, 32),
                log_prob_tensor_std=1.0,
                name="policy"
            )

    @variant(hide=True)
    def algo(self, setup, env, policy, mi_coeff, imitation_max_epochs, bonus_upscale_coeff):
        baseline = ZeroBaseline(env_spec=env.spec)

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
            batch_size=5000,
            n_itr=100,
            discount=0.99,
            gae_lambda=1.0,
            mi_coeff=mi_coeff,
            imitation_max_epochs=imitation_max_epochs,
            bonus_upscale_coeff=bonus_upscale_coeff,
        )
        yield algo


variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants:
    run_experiment_lite(
        v["algo"].train(),
        exp_prefix="0712-hrl-grid-2",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        mode="lab_kube",
        variant=v,
    )
