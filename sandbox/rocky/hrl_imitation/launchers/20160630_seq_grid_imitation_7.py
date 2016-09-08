


from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl_imitation.algos.fixed_clock_imitation import FixedClockImitation
from sandbox.rocky.hrl_imitation.approximate_posteriors.approximate_posterior import ApproximatePosterior
from sandbox.rocky.hrl_imitation.env_experts.seq_grid_expert import SeqGridExpert
from sandbox.rocky.hrl_imitation.policy_modules.seq_grid_policy_module1 import SeqGridPolicyModule

stub(globals())

"""
Test the simplest setup where the low-level policy only receives partial observation. Turns out that the approximate
posterior I had didn't have enough capacity, which is fixed now.
"""

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("seed", [11, 111, 211, 311, 411, 511, 611, 711, 811, 911])
vg.add("learning_rate", [1e-3])
vg.add("obs_feature_dim", [10])#, 2, 3, 5, 7, 10])
vg.add("feature_hidden_dim", [100])#, 32, 50, 100, 200])
vg.add("bottleneck_coeff", [0])

variants = vg.variants()

print("#Experiments:", len(variants))

for v in variants:
    env_expert = SeqGridExpert()
    recog = ApproximatePosterior(
        env_spec=env_expert.env_spec, subgoal_dim=4,
        subgoal_interval=3, obs_feature_dim=v["obs_feature_dim"],
        feature_hidden_dim=v["feature_hidden_dim"]
    )
    algo = FixedClockImitation(
        env_expert=env_expert,
        policy_module=SeqGridPolicyModule(),
        recog=recog,
        n_epochs=1000,
        n_sweep_per_epoch=5,
        learning_rate=v["learning_rate"],
        bottleneck_coeff=v["bottleneck_coeff"],
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="seq_grid_imitation_10",
        seed=v["seed"],
        variant=v,
        mode="local"
    )
    # break
