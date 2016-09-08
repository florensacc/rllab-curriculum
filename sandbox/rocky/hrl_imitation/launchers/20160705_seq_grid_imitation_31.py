


from rllab.misc.instrument import stub, run_experiment_lite

"""
Discrete bottleneck layer
"""

from sandbox.rocky.hrl_imitation.algos.fixed_clock_imitation17 import FixedClockImitation, SeqGridPolicyModule

stub(globals())
from rllab.misc.instrument import VariantGenerator

# import tensorflow as tf
# from rllab.misc import logger


# seed 11: doesn't work
vg = VariantGenerator()
vg.add("seed", [x * 100 + 11 for x in range(10)])  # 911])#11, 111, 211, 311, 411, 511, 611, 711, 811, 911])
vg.add("learning_rate", [1e-3])#1e-3, 7.5e-4, 5e-4, 2.5e-4, 1e-4])
vg.add("mi_coeff", [0., 0.001, 0.01, 0.1, 1., 10.])
vg.add("low_policy_obs", ['full'])#'full', 'partial'])
vg.add("dim_multiple", [1, 2, 5, 10, 20])

variants = vg.variants()

print("#Experiments:", len(variants))

iters = []
for v in variants:
    algo = FixedClockImitation(
        policy_module=SeqGridPolicyModule(low_policy_obs=v['low_policy_obs']),
        learning_rate=v["learning_rate"],
        mi_coeff=v["mi_coeff"],
        n_epochs=2000,
        bottleneck_dim=3 * v["dim_multiple"],
        subgoal_dim=4 * v["dim_multiple"],
        batch_size=100,
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="seq_grid_imitation_31",
        seed=v["seed"],
        variant=v,
        mode="lab_kube",
        snapshot_mode="last",
    )
    # break
