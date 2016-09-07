


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
    # Use both state and action, passed to a GRU
    ApproximatePosterior_orig,
    # Use only the action, passed to a GRU
    ApproximatePosterior0,
    # Use only the action, passed to an MLP
    ApproximatePosterior1,
    # Use both state and action, passed to an MLP
    ApproximatePosterior2,
])

"""
Observations:
- If the low-level policy receives the full observation:
    - AP_orig and AP_0 has I(a;g|s) near 0 when no bonus, and struggles to achieve high MI when bonus is given
    - AP_1 and AP_2 were able to achieve MI around 0.5 for some runs even without bonus, and does much better when
      bonus is given
- If the low-level policy only sees the action history:
    - The optimization of vlb is prone to local optima, and sometimes get stuck in non-optimal solutions. These
      solutions also correspond to lower MI.
    - The bonus term has almost no effect on the result. Even in this case, sometimes the MI still couldn't get well
      optimized because of the above issue.
    - The exact H(g|z) is positively correlated with how well the optimization is done, and also with the performance on
      the test environment. It is still questionable whether these terms are all necessary.
"""


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
