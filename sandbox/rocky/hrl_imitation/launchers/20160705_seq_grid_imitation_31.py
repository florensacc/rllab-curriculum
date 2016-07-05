from __future__ import absolute_import
from __future__ import print_function

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
vg.add("mi_coeff", [1.])#0., 0.001, 0.01, 0.1, 1., 10.])
vg.add("low_policy_obs", ['full'])#'full', 'partial'])

variants = vg.variants()

print("#Experiments:", len(variants))

iters = []
for v in variants:
    algo = FixedClockImitation(
        policy_module=SeqGridPolicyModule(low_policy_obs=v['low_policy_obs']),
        learning_rate=v["learning_rate"],
        mi_coeff=v["mi_coeff"],
        n_epochs=5000,
        bottleneck_dim=3,
        subgoal_dim=4,
        batch_size=100,
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="seq_grid_imitation_31",
        seed=v["seed"],
        variant=v,
        mode="local",
        snapshot_mode="last",
    )
    break
