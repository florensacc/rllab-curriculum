from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.hrl.policies.continuous_rnn_policy import ContinuousRNNPolicy
# from sandbox.rocky.hrl.algos.alt_bonus_algos import AltNPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from sandbox.rocky.tf.algos.trpo import TRPO#hrl.algos.tf_bonus_algos import BonusTRPO
from rllab.algos.ppo import PPO
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.envs.perm_grid_env import PermGridEnv
import sys

stub(globals())

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("grid_size", [5])  # , 7, 9, 11])
vg.add("batch_size", [20000])  # , 10000, 20000])
vg.add("seed", [11, 111, 211, 311, 411])

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    env = PermGridEnv(size=v["grid_size"], n_objects=v["grid_size"], object_seed=0)
    policy = ContinuousRNNPolicy(
        env_spec=env.spec,
        hidden_state_dim=1,#v["grid_size"],
        bottleneck_dim=5,
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = PPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v["batch_size"],
        step_size=0.01,
        max_path_length=100,
        n_itr=500,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="hier_cont_no_truncate",
        n_parallel=1,
        seed=v["seed"],
        mode="local",
    )
