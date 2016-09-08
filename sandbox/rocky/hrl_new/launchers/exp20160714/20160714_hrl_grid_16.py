


from sandbox.rocky.hrl_new.envs.image_grid_world import RandomImageGridWorld
from rllab.algos.trpo import TRPO
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp, PerlmutterHvp
#optimizers.import ConjugateGradientOptimizer, FiniteDifferenceHvp
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.core.network import MLP
from sandbox.rocky.hrl.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.baselines.zero_baseline import ZeroBaseline
import theano.tensor as TT

stub(globals())
from rllab.misc.instrument import VariantGenerator, variant

"""
Try RNN policies w/ Theano and tune optimizer
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [x * 100 + 11 for x in range(5)]

    @variant
    def setup(self):
        return ["gru_policy", "flat_policy"]#, "gru_policy"]#, "flat_policy"]

    @variant
    def env_size(self):
        return [5]

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
        env = CompoundActionSequenceEnv(wrapped_env, action_map, obs_include_history=True)
        yield env

    @variant
    def has_feature_network(self, setup):
        if setup == "flat_policy":
            return [None]
        return [True]#, False]

    @variant(hide=True)
    def policy(self, setup, env, has_feature_network):
        if setup == "flat_policy":
            yield CategoricalMLPPolicy(
                env_spec=env.spec,
            )
        else:
            if has_feature_network:
                obs_dim = env.spec.observation_space.flat_dim
                action_dim = env.spec.action_space.flat_dim
                input_dim = obs_dim + action_dim
                feature_network = MLP(
                    name="feature_network",
                    input_shape=(input_dim,),
                    hidden_nonlinearity=TT.tanh,
                    output_nonlinearity=TT.tanh,
                    hidden_sizes=(100,),
                    output_dim=100,
                )
            else:
                feature_network = None
            yield CategoricalGRUPolicy(
                env_spec=env.spec,
                state_include_action=True,
                feature_network=feature_network,
            )

    @variant
    def hvp_approach(self):
        return [PerlmutterHvp]#FiniteDifferenceHvp, PerlmutterHvp]

    @variant
    def base_eps(self, hvp_approach):
        if hvp_approach == FiniteDifferenceHvp:
            return [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        return [None]

    @variant
    def symmetric(self, hvp_approach):
        if hvp_approach == FiniteDifferenceHvp:
            return [True, False]
        return [None]

    @variant(hide=True)
    def algo(self, env, policy, hvp_approach, base_eps, symmetric):
        baseline = ZeroBaseline(env_spec=env.spec)
        if hvp_approach == FiniteDifferenceHvp:
            hvp_approach = FiniteDifferenceHvp(base_eps=base_eps, symmetric=symmetric)
        elif hvp_approach == PerlmutterHvp:
            hvp_approach = PerlmutterHvp()
        else:
            raise NotImplementedError

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            batch_size=5000,
            n_itr=200,
            discount=0.99,
            gae_lambda=1.0,
            optimizer=ConjugateGradientOptimizer(hvp_approach=hvp_approach)
        )
        yield algo


variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants:
    run_experiment_lite(
        v["algo"].train(),
        exp_prefix="0714-hrl-grid-16",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        mode="lab_kube",
        variant=v,
    )
    # break
