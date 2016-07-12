from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.hrl.envs.atari import AtariEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite

"""
Even smaller discount factors
"""

stub(globals())
from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("seed", [x * 100 + 11 for x in range(5)])
vg.add("hidden_sizes", [(32, 32)])#, (64, 64), (256, 256)])
vg.add("discount", [0.9, 0.95, 0.99])#, 0.999, 1.0])
vg.add("gae_lambda", [1., 0.99, 0.95])
vg.add("frame_skip", [4])#, 12])

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    env = TfEnv(AtariEnv(game="seaquest", obs_type="ram", frame_skip=v["frame_skip"]))
    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        name="policy",
        hidden_sizes=v["hidden_sizes"],
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        max_path_length=18000 / v["frame_skip"],
        batch_size=50000 * 4 / v["frame_skip"],
        discount=v["discount"],
        gae_lambda=v["gae_lambda"]
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="0708-trpo-atari-ram",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        variant=v,
        mode="lab_kube",
    )
