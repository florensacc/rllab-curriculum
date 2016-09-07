


from rllab.misc.instrument import stub, run_experiment_lite

"""
Do not include obstacles etc. in the map observation
"""

from sandbox.rocky.hrl_new.launchers.exp20160721.fixed_clock_imitation23 import FixedClockImitation, SeqGridPolicyModule

stub(globals())
from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("seed", [x * 100 + 11 for x in range(10)])
vg.add("learning_rate", [1e-3])#1e-3, 7.5e-4, 5e-4, 2.5e-4, 1e-4])
vg.add("mi_coeff", [0., 0.1, 1., 10.])
vg.add("kl_coeff", [0.1, 1., 10.])
vg.add("low_policy_obs", ['full'])#'full', 'partial'])
vg.add("log_prob_tensor_std", [10.0])#0.01, 0.1, 1.0, 10.0])
vg.add("batch_size", [32])#, 500, 6000])
vg.add("dim_multiple", [1, 5])#, 20])#1, 5, 20])

variants = vg.variants()

print("#Experiments:", len(variants))

iters = []
for v in variants:
    algo = FixedClockImitation(
        policy_module=SeqGridPolicyModule(low_policy_obs=v['low_policy_obs'], log_prob_tensor_std=v['log_prob_tensor_std']),
        learning_rate=v["learning_rate"],
        mi_coeff=v["mi_coeff"],
        n_epochs=2000,
        bottleneck_dim=3 * v["dim_multiple"],
        subgoal_dim=4 * v["dim_multiple"],
        batch_size=v["batch_size"],
        kl_coeff=v["kl_coeff"],
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="seq-grid-imitation-41",
        seed=v["seed"],
        variant=v,
        mode="lab_kube",
        snapshot_mode="last",
    )
