from __future__ import absolute_import
from __future__ import print_function

from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.hrl_imitation.approximate_posteriors.approximate_posterior import ApproximatePosterior

"""
mutual information bonus with stochastic bottleneck
"""

# if False:
from sandbox.rocky.hrl_imitation.algos.fixed_clock_imitation2 import FixedClockImitation
from sandbox.rocky.hrl_imitation.env_experts.seq_grid_expert2 import SeqGridExpert
from sandbox.rocky.hrl_imitation.policy_modules.seq_grid_policy_module3 import SeqGridPolicyModule
# else:
#     from sandbox.rocky.hrl_imitation.algos.fixed_clock_imitation import FixedClockImitation
#     from sandbox.rocky.hrl_imitation.env_experts.seq_grid_expert import SeqGridExpert
#     from sandbox.rocky.hrl_imitation.policy_modules.seq_grid_policy_module1 import SeqGridPolicyModule

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("seed", [11, 111, 211, 311, 411])
vg.add("learning_rate", [1e-3])

variants = vg.variants()

print("#Experiments:", len(variants))

for v in variants:
    env_expert = SeqGridExpert()
    recog = ApproximatePosterior(env_spec=env_expert.env_spec, subgoal_dim=4,
                                 subgoal_interval=3)
    algo = FixedClockImitation(
        env_expert=env_expert,
        policy_module=SeqGridPolicyModule(),
        recog=recog,
        learning_rate=v["learning_rate"],
        n_sweep_per_epoch=5,
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="seq_grid_imitation_5",
        seed=v["seed"],
        variant=v,
        mode="local"
    )
    break
