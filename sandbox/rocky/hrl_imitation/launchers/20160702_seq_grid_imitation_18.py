from __future__ import absolute_import
from __future__ import print_function

from rllab.misc.instrument import stub, run_experiment_lite

"""
Compare different policy representations.
"""

from sandbox.rocky.hrl_imitation.algos.fixed_clock_imitation5 import FixedClockImitation, SeqGridPolicyModule,\
    SeqGridPolicyModule1
from sandbox.rocky.hrl_imitation.algos.fixed_clock_imitation5 import ApproximatePosterior as ApproximatePosterior_orig
from sandbox.rocky.hrl_imitation.approximate_posteriors.approximate_posterior import ApproximatePosterior as \
    ApproximatePosterior0
from sandbox.rocky.hrl_imitation.approximate_posteriors.approximate_posterior1 import ApproximatePosterior as \
    ApproximatePosterior1
from sandbox.rocky.hrl_imitation.approximate_posteriors.approximate_posterior2 import ApproximatePosterior as \
    ApproximatePosterior2

stub(globals())
from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("seed", [x * 100 + 11 for x in range(10)])
vg.add("learning_rate", [1e-3])
vg.add("mi_coeff", [0, 10.])
vg.add("ent_g_given_z_coeff", [0.])#, 10.])
vg.add("policy_module", [
    SeqGridPolicyModule,
    SeqGridPolicyModule1,
])
vg.add("recog", [
    ApproximatePosterior_orig,
    ApproximatePosterior0,
    ApproximatePosterior1,
    ApproximatePosterior2,
])

variants = vg.variants()

print("#Experiments:", len(variants))

iters = []
for v in variants:
    algo = FixedClockImitation(
        learning_rate=v["learning_rate"],
        approximate_posterior_cls=v["recog"],
        policy_module_cls=v["policy_module"],
        n_sweep_per_epoch=5,
        mi_coeff=v["mi_coeff"],
        ent_g_given_z_coeff=v["ent_g_given_z_coeff"],
        n_epochs=100,
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="seq_grid_imitation_18",
        seed=v["seed"],
        variant=v,
        mode="lab_kube",
        snapshot_mode="last",
    )
