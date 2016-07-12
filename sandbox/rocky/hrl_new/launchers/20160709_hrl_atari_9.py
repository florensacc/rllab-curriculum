from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.hrl.envs.atari import AtariEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.straw.optimizers.tf_conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from sandbox.rocky.hrl_new.optimizers.quadratic_penalty_optimizer import QuadraticPenaltyOptimizer
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

stub(globals())

from rllab.misc.instrument import VariantGenerator

"""
Play with optimization
"""

vg = VariantGenerator()
vg.add("seed", [x * 100 + 11 for x in range(5)])
vg.add("first_n_epoch", [1, 5, 10])

variants = vg.variants()

print("#Experiments: %d" % len(variants))



for v in variants:

    env = TfEnv(AtariEnv(game="seaquest", obs_type="ram", frame_skip=4))
    policy = CategoricalMLPPolicy(env_spec=env.spec, name="policy")
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        max_path_length=4500,
        batch_size=50000,
        discount=0.99,
        gae_lambda=0.99,
        optimizer=QuadraticPenaltyOptimizer(
            epoch_seq=[v["first_n_epoch"], 5, 5, 5, 5],
            batch_size_seq=[128, None, None, None, None],
            penalty_seq=[10, 100, 1000, 10000, 100000]
        ),
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="0709-hrl-atari-ram-9",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        variant=v,
        mode="lab_kube",
    )
