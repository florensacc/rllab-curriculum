from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.hrl_new.algos.hrl_algos import HierTRPO
from sandbox.rocky.hrl.envs.atari import AtariEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.hrl_new.policies.fixed_clock_policy import FixedClockPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.straw.optimizers.tf_conjugate_gradient_optimizer import ConjugateGradientOptimizer, \
    FiniteDifferenceHvp, PerlmutterHvp
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

stub(globals())
from rllab.misc.instrument import VariantGenerator

"""
More extensive test with more games
"""

vg = VariantGenerator()
vg.add("seed", [x * 100 + 11 for x in range(5)])
vg.add("game", ["seaquest", "breakout", "frostbite", "space_invaders", "freeway"])
vg.add("is_hierarchical", [True, False])  # 10, 50])#], 50])
vg.add("subgoal_dim", [50])  # 10, 50])#], 50])
vg.add("bottleneck_dim", [50])  # 10, 50])
vg.add("subgoal_interval", lambda is_hierarchical: [1, 3, 10] if is_hierarchical else [None])
vg.add("hvp_cls", lambda is_hierarchical: [FiniteDifferenceHvp] if is_hierarchical else [FiniteDifferenceHvp,
                                                                                         PerlmutterHvp])
vg.add("hidden_sizes", [(32, 32)])  # , (64, 64), (256, 256)])
vg.add("log_prob_tensor_std", [1.0])  # , 10.0, 0.1, 0.01])
vg.add("base_eps", lambda hvp_cls: [1e-8, 1e-5] if hvp_cls == FiniteDifferenceHvp else [None])

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    env = TfEnv(AtariEnv(game=v["game"], obs_type="ram", frame_skip=4))
    if v["is_hierarchical"]:
        policy = FixedClockPolicy(
            env_spec=env.spec,
            subgoal_dim=v["subgoal_dim"],
            bottleneck_dim=v["bottleneck_dim"],
            subgoal_interval=v["subgoal_interval"],
            hidden_sizes=v["hidden_sizes"],
            log_prob_tensor_std=v["log_prob_tensor_std"],
        )
    else:
        policy = CategoricalMLPPolicy(env_spec=env.spec, name="policy")

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    if v["hvp_cls"] == FiniteDifferenceHvp:
        continue
    else:
        hvp_approach = PerlmutterHvp()

    if v["is_hierarchical"]:
        algo = HierTRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=4500,
            batch_size=50000,
            discount=0.99,
            gae_lambda=0.99,
            mi_coeff=0.,
            optimizer=ConjugateGradientOptimizer(hvp_approach=hvp_approach)
        )
    else:
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=4500,
            batch_size=50000,
            discount=0.99,
            gae_lambda=0.99,
            optimizer=ConjugateGradientOptimizer(hvp_approach=hvp_approach)
        )

    run_experiment_lite(
        algo.train(),
        exp_prefix="0709-hrl-atari-ram-10-1",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        variant=v,
        mode="lab_kube",
    )
